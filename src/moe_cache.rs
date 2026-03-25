//! Expert residency management for offloaded MoE weights (the C++
//! `MoEManager`, `LRU_MoEManager` and `MRU_MoEManager`).
//!
//! When the experts do not all fit in device memory, most of them live in
//! host memory and are copied in on demand into a fixed number of device
//! slots. The cache decides which resident expert to evict when a new one
//! is needed; the actual copies are done by a [`Residency`] implementation
//! (asynchronous host-to-device transfers on the CUDA backend, a mock in
//! tests).
//!
//! Policies:
//! * [`Policy::Lru`]: evict the least recently used expert.
//! * [`Policy::LayerDistance`]: evict the expert whose layer comes up
//!   *last* in the cyclic layer order as seen from the layer being loaded,
//!   skipping protected layers (the C++ "MRU" manager). Layers run in order
//!   every token, so the expert needed furthest in the future goes first.

use std::collections::HashMap;

use crate::error::Result;

/// `(layer, expert)`
pub type ExpertKey = (usize, usize);

/// Moves expert weights between host and device.
pub trait Residency {
    /// Start loading `key` into the device slot freed by `reuse` (if any).
    fn load(&mut self, key: ExpertKey, reuse: Option<ExpertKey>) -> Result<()>;
    /// Whether a load started for `key` has completed.
    fn is_ready(&self, key: ExpertKey) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Lru,
    LayerDistance { num_layers: usize },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

const NIL: usize = usize::MAX;

#[derive(Debug, Clone)]
struct Node {
    key: ExpertKey,
    prev: usize,
    next: usize,
}

/// Fixed-capacity expert cache. Experts registered as pinned are always
/// resident and never counted against the capacity (the C++ engine keeps
/// the first layers' experts on the GPU this way).
pub struct ExpertCache<R: Residency> {
    residency: R,
    policy: Policy,
    capacity: usize,
    /// slab of list nodes; `head` = most recently used
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    head: usize,
    tail: usize,
    /// resident (non-pinned) expert -> node index
    index: HashMap<ExpertKey, usize>,
    pinned: HashMap<ExpertKey, ()>,
    known: HashMap<ExpertKey, ()>,
    stats: CacheStats,
}

impl<R: Residency> ExpertCache<R> {
    pub fn new(residency: R, capacity: usize, policy: Policy) -> Self {
        ExpertCache {
            residency,
            policy,
            capacity,
            nodes: Vec::new(),
            free_nodes: Vec::new(),
            head: NIL,
            tail: NIL,
            index: HashMap::new(),
            pinned: HashMap::new(),
            known: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Make an expert known to the cache (the C++ `inform`).
    pub fn register(&mut self, key: ExpertKey, pinned: bool) {
        self.known.insert(key, ());
        if pinned {
            self.pinned.insert(key, ());
        }
    }

    pub fn residency(&self) -> &R {
        &self.residency
    }

    pub fn residency_mut(&mut self) -> &mut R {
        &mut self.residency
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn is_resident(&self, key: ExpertKey) -> bool {
        self.pinned.contains_key(&key) || self.index.contains_key(&key)
    }

    /// Resident and done loading (the C++ `is_done`).
    pub fn is_ready(&self, key: ExpertKey) -> bool {
        self.pinned.contains_key(&key) || (self.index.contains_key(&key) && self.residency.is_ready(key))
    }

    /// Ensure `key` is (being) loaded and mark it as just used. Experts from
    /// `protected_layers` are not evicted by the layer-distance policy.
    pub fn prepare(&mut self, key: ExpertKey, protected_layers: &[usize]) -> Result<()> {
        if !self.known.contains_key(&key) || self.pinned.contains_key(&key) {
            return Ok(());
        }
        if let Some(&n) = self.index.get(&key) {
            self.stats.hits += 1;
            self.unlink(n);
            self.push_front(n);
            return Ok(());
        }
        self.stats.misses += 1;
        let mut reuse = None;
        if self.index.len() >= self.capacity && self.tail != NIL {
            let victim = self.pick_victim(key, protected_layers);
            let vkey = self.nodes[victim].key;
            self.unlink(victim);
            self.index.remove(&vkey);
            self.free_nodes.push(victim);
            self.stats.evictions += 1;
            reuse = Some(vkey);
        }
        let n = self.alloc_node(key);
        self.push_front(n);
        self.index.insert(key, n);
        self.residency.load(key, reuse)
    }

    fn pick_victim(&self, incoming: ExpertKey, protected_layers: &[usize]) -> usize {
        match self.policy {
            Policy::Lru => self.tail,
            Policy::LayerDistance { num_layers } => {
                // Furthest upcoming unprotected layer; ties go to the least
                // recently used. Unlike the C++ manager, an unprotected
                // expert from the current layer beats evicting a protected
                // one; only if everything is protected do we fall back to LRU.
                let mut best: Option<(usize, usize)> = None;
                let mut n = self.tail;
                while n != NIL {
                    let layer = self.nodes[n].key.0;
                    if !protected_layers.contains(&layer) {
                        let dist = (layer + num_layers - incoming.0 % num_layers) % num_layers;
                        if best.is_none_or(|(_, d)| dist > d) {
                            best = Some((n, dist));
                        }
                    }
                    n = self.nodes[n].prev;
                }
                best.map_or(self.tail, |(n, _)| n)
            }
        }
    }

    fn alloc_node(&mut self, key: ExpertKey) -> usize {
        let node = Node { key, prev: NIL, next: NIL };
        match self.free_nodes.pop() {
            Some(i) => {
                self.nodes[i] = node;
                i
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        }
    }

    fn unlink(&mut self, n: usize) {
        let (p, x) = (self.nodes[n].prev, self.nodes[n].next);
        if p != NIL {
            self.nodes[p].next = x;
        } else {
            self.head = x;
        }
        if x != NIL {
            self.nodes[x].prev = p;
        } else {
            self.tail = p;
        }
        self.nodes[n].prev = NIL;
        self.nodes[n].next = NIL;
    }

    fn push_front(&mut self, n: usize) {
        self.nodes[n].prev = NIL;
        self.nodes[n].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = n;
        }
        self.head = n;
        if self.tail == NIL {
            self.tail = n;
        }
    }

    /// Resident experts from most to least recently used.
    pub fn recency_order(&self) -> Vec<ExpertKey> {
        let mut out = Vec::with_capacity(self.index.len());
        let mut n = self.head;
        while n != NIL {
            out.push(self.nodes[n].key);
            n = self.nodes[n].next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records every load; loads complete when `finish` is called.
    #[derive(Default)]
    struct Mock {
        loads: Vec<(ExpertKey, Option<ExpertKey>)>,
        in_flight: Vec<ExpertKey>,
    }

    impl Residency for Mock {
        fn load(&mut self, key: ExpertKey, reuse: Option<ExpertKey>) -> Result<()> {
            self.loads.push((key, reuse));
            self.in_flight.push(key);
            Ok(())
        }
        fn is_ready(&self, key: ExpertKey) -> bool {
            !self.in_flight.contains(&key)
        }
    }

    fn cache(capacity: usize, policy: Policy) -> ExpertCache<Mock> {
        let mut c = ExpertCache::new(Mock::default(), capacity, policy);
        for layer in 0..4 {
            for e in 0..8 {
                c.register((layer, e), layer == 0);
            }
        }
        c
    }

    #[test]
    fn lru_evicts_least_recent_and_reuses_its_slot() {
        let mut c = cache(2, Policy::Lru);
        c.prepare((1, 0), &[]).unwrap();
        c.prepare((1, 1), &[]).unwrap();
        c.prepare((1, 0), &[]).unwrap(); // hit, (1,1) is now LRU
        c.prepare((2, 5), &[]).unwrap();
        assert_eq!(c.recency_order(), vec![(2, 5), (1, 0)]);
        assert_eq!(c.residency().loads.last(), Some(&((2, 5), Some((1, 1)))));
        assert_eq!(c.stats(), CacheStats { hits: 1, misses: 3, evictions: 1 });
    }

    #[test]
    fn pinned_experts_are_free_and_ready() {
        let mut c = cache(1, Policy::Lru);
        c.prepare((0, 3), &[]).unwrap();
        assert!(c.is_ready((0, 3)));
        assert!(c.residency().loads.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn readiness_follows_the_transfer() {
        let mut c = cache(4, Policy::Lru);
        c.prepare((3, 1), &[]).unwrap();
        assert!(c.is_resident((3, 1)) && !c.is_ready((3, 1)));
        c.residency_mut().in_flight.clear();
        assert!(c.is_ready((3, 1)));
        assert!(!c.is_ready((3, 2)));
    }

    #[test]
    fn layer_distance_evicts_the_furthest_upcoming_layer() {
        let mut c = cache(3, Policy::LayerDistance { num_layers: 4 });
        c.prepare((1, 0), &[]).unwrap();
        c.prepare((2, 0), &[]).unwrap();
        c.prepare((3, 0), &[]).unwrap();
        c.prepare((1, 0), &[]).unwrap(); // most recent now; LRU would evict (2, 0)
        // loading for layer 2: distances 1 -> 3, 2 -> 0, 3 -> 1
        c.prepare((2, 7), &[]).unwrap();
        assert!(!c.is_resident((1, 0)));
        assert!(c.is_resident((2, 0)));
        // from layer 3 with layer 2 protected only (3, 0) is evictable
        c.prepare((3, 4), &[2]).unwrap();
        assert!(!c.is_resident((3, 0)));
        assert!(c.is_resident((2, 0)) && c.is_resident((2, 7)));
        // everything protected: fall back to the LRU tail
        c.prepare((1, 1), &[2, 3]).unwrap();
        assert_eq!(c.residency().loads.last(), Some(&((1, 1), Some((2, 0)))));
    }
}
