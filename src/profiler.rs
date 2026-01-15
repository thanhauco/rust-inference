//! Lightweight timing and energy measurement (the C++ `profiler.h`).
//!
//! Section timers are global atomic counters, off by default; `enable(true)`
//! (the runner's `--profile`) turns them on. The `profile!` macro times one
//! expression into a section. `PowerProfiler` reads cumulative energy
//! counters (Intel RAPL for the CPU package, NVML for the GPU when the CUDA
//! backend is built) around a region of work.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Section {
    Embedding,
    Rope,
    QkvProjection,
    QkNorm,
    PosEmbedding,
    KvAppend,
    Attention,
    OutProjection,
    Norm,
    Mlp,
    Router,
    Experts,
    ExpertScatter,
    LmHead,
    Sampling,
}

impl Section {
    pub const ALL: [Section; 15] = [
        Section::Embedding,
        Section::Rope,
        Section::QkvProjection,
        Section::QkNorm,
        Section::PosEmbedding,
        Section::KvAppend,
        Section::Attention,
        Section::OutProjection,
        Section::Norm,
        Section::Mlp,
        Section::Router,
        Section::Experts,
        Section::ExpertScatter,
        Section::LmHead,
        Section::Sampling,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Section::Embedding => "embedding",
            Section::Rope => "rope tables",
            Section::QkvProjection => "attn q/k/v proj",
            Section::QkNorm => "attn q/k norm",
            Section::PosEmbedding => "attn rope",
            Section::KvAppend => "kv cache append",
            Section::Attention => "attention",
            Section::OutProjection => "attn o proj",
            Section::Norm => "rmsnorm",
            Section::Mlp => "dense mlp",
            Section::Router => "moe router",
            Section::Experts => "moe experts",
            Section::ExpertScatter => "moe gather/scatter",
            Section::LmHead => "lm head",
            Section::Sampling => "sampling",
        }
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static MICROS: [AtomicU64; Section::ALL.len()] = [const { AtomicU64::new(0) }; Section::ALL.len()];
static CALLS: [AtomicU64; Section::ALL.len()] = [const { AtomicU64::new(0) }; Section::ALL.len()];

pub fn enable(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn record(section: Section, d: Duration) {
    MICROS[section as usize].fetch_add(d.as_micros() as u64, Ordering::Relaxed);
    CALLS[section as usize].fetch_add(1, Ordering::Relaxed);
}

pub fn reset() {
    for (m, c) in MICROS.iter().zip(&CALLS) {
        m.store(0, Ordering::Relaxed);
        c.store(0, Ordering::Relaxed);
    }
}

/// `(section, total time, calls)` for every section that ran.
pub fn snapshot() -> Vec<(Section, Duration, u64)> {
    Section::ALL
        .iter()
        .map(|&s| {
            let us = MICROS[s as usize].load(Ordering::Relaxed);
            (s, Duration::from_micros(us), CALLS[s as usize].load(Ordering::Relaxed))
        })
        .filter(|(_, _, calls)| *calls > 0)
        .collect()
}

/// Human-readable breakdown, largest first.
pub fn report() -> String {
    let mut rows = snapshot();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let total: Duration = rows.iter().map(|r| r.1).sum();
    let mut out = String::from("section                  total(s)   share    calls\n");
    for (s, d, calls) in rows {
        let share = if total.is_zero() { 0.0 } else { 100.0 * d.as_secs_f64() / total.as_secs_f64() };
        out.push_str(&format!("{:<22} {:>10.4} {:>6.1}% {:>8}\n", s.name(), d.as_secs_f64(), share, calls));
    }
    out
}

/// Time `$e` into `$section` when profiling is enabled.
#[macro_export]
macro_rules! profile {
    ($section:expr, $e:expr) => {{
        if $crate::profiler::enabled() {
            let t0 = ::std::time::Instant::now();
            let r = $e;
            $crate::profiler::record($section, t0.elapsed());
            r
        } else {
            $e
        }
    }};
}

// ---------------------------------------------------------------------------
// energy
// ---------------------------------------------------------------------------

const RAPL_ENERGY: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";
const RAPL_MAX: &str = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// A cumulative energy counter in joules.
pub trait EnergyCounter {
    fn name(&self) -> &str;
    /// Joules consumed between two readings of this counter.
    fn read(&mut self) -> Option<f64>;
}

/// CPU package energy from Intel RAPL (needs read access to powercap).
pub struct Rapl {
    max_uj: u64,
    last_uj: u64,
    total_j: f64,
}

impl Rapl {
    pub fn open() -> Option<Self> {
        let last_uj = read_u64(RAPL_ENERGY)?;
        Some(Rapl { max_uj: read_u64(RAPL_MAX).unwrap_or(u64::MAX), last_uj, total_j: 0.0 })
    }
}

impl EnergyCounter for Rapl {
    fn name(&self) -> &str {
        "cpu"
    }

    fn read(&mut self) -> Option<f64> {
        let now = read_u64(RAPL_ENERGY)?;
        let delta = if now >= self.last_uj { now - self.last_uj } else { self.max_uj - self.last_uj + now };
        self.last_uj = now;
        self.total_j += delta as f64 / 1e6;
        Some(self.total_j)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PowerReading {
    pub seconds: f64,
    /// joules per counter, e.g. `[("cpu", 12.3), ("gpu", 80.1)]`
    pub joules: Vec<(String, f64)>,
}

impl PowerReading {
    pub fn total_joules(&self) -> f64 {
        self.joules.iter().map(|j| j.1).sum()
    }

    pub fn watts(&self) -> f64 {
        if self.seconds > 0.0 { self.total_joules() / self.seconds } else { 0.0 }
    }
}

impl std::fmt::Display for PowerReading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Joules: {:.2}J Time: {:.3}s {:.2}W", self.total_joules(), self.seconds, self.watts())?;
        for (name, j) in &self.joules {
            write!(f, " [{name} {j:.2}J]")?;
        }
        Ok(())
    }
}

/// Measures energy over a region, like the C++ RAII `PowerProfiler` (which
/// also sleeps first so the previous phase's power draw settles).
pub struct PowerProfiler {
    counters: Vec<Box<dyn EnergyCounter>>,
    start: Vec<f64>,
    t0: Instant,
}

impl PowerProfiler {
    pub fn start(mut counters: Vec<Box<dyn EnergyCounter>>, settle: Duration) -> Self {
        std::thread::sleep(settle);
        let start = counters.iter_mut().map(|c| c.read().unwrap_or(0.0)).collect();
        PowerProfiler { counters, start, t0: Instant::now() }
    }

    pub fn finish(mut self) -> (PowerReading, Vec<Box<dyn EnergyCounter>>) {
        let seconds = self.t0.elapsed().as_secs_f64();
        let joules = self
            .counters
            .iter_mut()
            .zip(&self.start)
            .map(|(c, s)| (c.name().to_string(), c.read().map_or(0.0, |v| v - s)))
            .collect();
        (PowerReading { seconds, joules }, self.counters)
    }
}

/// Every energy counter available on this machine.
pub fn available_counters() -> Vec<Box<dyn EnergyCounter>> {
    let mut v: Vec<Box<dyn EnergyCounter>> = Vec::new();
    if let Some(r) = Rapl::open() {
        v.push(Box::new(r));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(f64);
    impl EnergyCounter for Fake {
        fn name(&self) -> &str {
            "fake"
        }
        fn read(&mut self) -> Option<f64> {
            self.0 += 5.0;
            Some(self.0)
        }
    }

    #[test]
    fn power_reading_uses_deltas() {
        let p = PowerProfiler::start(vec![Box::new(Fake(100.0))], Duration::ZERO);
        let (r, _) = p.finish();
        assert_eq!(r.joules, vec![("fake".to_string(), 5.0)]);
        assert!(r.seconds >= 0.0);
    }

    #[test]
    fn sections_accumulate() {
        enable(true);
        let v = profile!(Section::Sampling, 2 + 2);
        assert_eq!(v, 4);
        record(Section::Sampling, Duration::from_millis(3));
        let snap = snapshot();
        let (_, d, calls) = snap.iter().find(|r| r.0 == Section::Sampling).unwrap();
        assert!(*calls >= 2 && *d >= Duration::from_millis(3));
        assert!(report().contains("sampling"));
    }
}
