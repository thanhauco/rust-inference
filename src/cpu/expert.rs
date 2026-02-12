//! CPU execution of MoE experts (the C++ `fp8_expert_cpu`).
//!
//! For one expert and T tokens:
//!
//! ```text
//! h = up(x) * silu(gate(x))      [T, I]
//! y = down(h)                    [T, H]
//! ```
//!
//! Each row of a weight matrix is decoded once and dotted against up to four
//! tokens at a time (`linear::row_dot`), so dispatching several tokens to an
//! expert costs about one pass over its weights. With `bf16_rounding` the
//! intermediate and output are rounded exactly where the CUDA fused kernel
//! rounds (`bf16(up) * bf16(silu(gate))`, bf16 output), so CPU-computed
//! experts are interchangeable with GPU-computed ones.
//!
//! `ExpertExecutor` is a persistent pool. Every submitted expert is split into
//! row chunks over two dependent stages (gate/up rows of I, then down rows of
//! H) and workers steal chunks across *all* in-flight experts, so both "8
//! experts x 1 token" and "1 expert x many tokens" use every core. Submission
//! is non-blocking: the caller can poll `ExpertJob::is_done` while doing other
//! work (the CUDA backend overlaps it with GPU experts).

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

use super::linear::{Weight, row_dot};
use crate::dtype;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Rows per scheduling chunk.
const CHUNK_ROWS: usize = 64;

/// One expert's three projections (any dtype the linear kernels accept).
#[derive(Clone)]
pub struct ExpertWeights {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

impl ExpertWeights {
    pub fn new(gate: Tensor, up: Tensor, down: Tensor) -> Result<Self> {
        let (i, h) = (gate.dim(0), gate.dim(1));
        if gate.ndim() != 2 || up.shape() != gate.shape() || down.shape() != [h, i] {
            return Err(Error::Shape(format!(
                "expert gate {:?} up {:?} down {:?}",
                gate.shape(),
                up.shape(),
                down.shape()
            )));
        }
        Ok(ExpertWeights { gate, up, down })
    }

    pub fn hidden(&self) -> usize {
        self.gate.dim(1)
    }

    pub fn intermediate(&self) -> usize {
        self.gate.dim(0)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn swiglu(gate: f32, up: f32, bf16: bool) -> f32 {
    if bf16 { dtype::round_bf16(dtype::round_bf16(up) * dtype::round_bf16(silu(gate))) } else { up * silu(gate) }
}

/// Run `f` over token tiles of 4 (with a 1-token remainder).
fn for_token_tiles(t: usize, mut f4: impl FnMut(usize), mut f1: impl FnMut(usize)) {
    let mut i = 0;
    while i + 4 <= t {
        f4(i);
        i += 4;
    }
    while i < t {
        f1(i);
        i += 1;
    }
}

/// Stage 1 for rows `n0..n1` of I: `h[t][n] = swiglu(gate[n].x_t, up[n].x_t)`.
///
/// # Safety
/// `h` must point to a `[T, I]` buffer; no one else may touch columns
/// `n0..n1` concurrently.
unsafe fn gateup_rows(
    w: &ExpertWeights,
    x: &[f32],
    t: usize,
    h: *mut f32,
    n0: usize,
    n1: usize,
    bf16: bool,
) -> Result<()> {
    let (hd, id) = (w.hidden(), w.intermediate());
    let (gate, up) = (Weight::new(&w.gate)?, Weight::new(&w.up)?);
    let xr = |i: usize| &x[i * hd..(i + 1) * hd];
    for n in n0..n1 {
        let (gr, ur) = (gate.row(n), up.row(n));
        for_token_tiles(
            t,
            |i| {
                let xs = [xr(i), xr(i + 1), xr(i + 2), xr(i + 3)];
                let (g, u) = (row_dot::<4>(gr, &xs), row_dot::<4>(ur, &xs));
                for j in 0..4 {
                    // SAFETY: (i + j, n) is inside [T, I] and column n is ours.
                    unsafe { *h.add((i + j) * id + n) = swiglu(g[j], u[j], bf16) };
                }
            },
            |i| {
                let xs = [xr(i)];
                let (g, u) = (row_dot::<1>(gr, &xs), row_dot::<1>(ur, &xs));
                unsafe { *h.add(i * id + n) = swiglu(g[0], u[0], bf16) };
            },
        );
    }
    Ok(())
}

/// Stage 2 for rows `n0..n1` of H: `y[t][n] = down[n].h_t`.
///
/// # Safety
/// As for `gateup_rows`, with `y: [T, H]`.
unsafe fn down_rows(
    w: &ExpertWeights,
    h: &[f32],
    t: usize,
    y: *mut f32,
    n0: usize,
    n1: usize,
    bf16: bool,
) -> Result<()> {
    let (hd, id) = (w.hidden(), w.intermediate());
    let down = Weight::new(&w.down)?;
    let hr = |i: usize| &h[i * id..(i + 1) * id];
    let out = |v: f32| if bf16 { dtype::round_bf16(v) } else { v };
    for n in n0..n1 {
        let dr = down.row(n);
        for_token_tiles(
            t,
            |i| {
                let r = row_dot::<4>(dr, &[hr(i), hr(i + 1), hr(i + 2), hr(i + 3)]);
                for (j, v) in r.into_iter().enumerate() {
                    // SAFETY: (i + j, n) is inside [T, H] and column n is ours.
                    unsafe { *y.add((i + j) * hd + n) = out(v) };
                }
            },
            |i| unsafe { *y.add(i * hd + n) = out(row_dot::<1>(dr, &[hr(i)])[0]) },
        );
    }
    Ok(())
}

/// Whole expert on the calling thread.
pub fn expert_ffn(w: &ExpertWeights, x: &[f32], tokens: usize, bf16_rounding: bool) -> Result<Vec<f32>> {
    check_input(w, x, tokens)?;
    let mut h = vec![0f32; tokens * w.intermediate()];
    let mut y = vec![0f32; tokens * w.hidden()];
    // SAFETY: single-threaded, buffers sized above.
    unsafe {
        gateup_rows(w, x, tokens, h.as_mut_ptr(), 0, w.intermediate(), bf16_rounding)?;
        down_rows(w, &h, tokens, y.as_mut_ptr(), 0, w.hidden(), bf16_rounding)?;
    }
    Ok(y)
}

/// f64 scalar reference (ground truth for tests).
pub fn expert_ffn_ref(w: &ExpertWeights, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
    check_input(w, x, tokens)?;
    let deq = |t: &Tensor| -> Result<Vec<f64>> {
        let (n, k) = (t.dim(0), t.dim(1));
        let vals = t.to_f32_vec()?;
        let mut out: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        if let Some(s) = t.scale() {
            let l = super::linear::BlockLayout::new(n, k, s.shape())?;
            let sv = s.as_slice::<f32>();
            for r in 0..n {
                for c in 0..k {
                    out[r * k + c] *= sv[(r / l.block_n) * l.scale_cols + c / l.block_k] as f64;
                }
            }
        }
        Ok(out)
    };
    let (g, u, d) = (deq(&w.gate)?, deq(&w.up)?, deq(&w.down)?);
    let (hd, id) = (w.hidden(), w.intermediate());
    let mut y = vec![0f32; tokens * hd];
    for t in 0..tokens {
        let xt = &x[t * hd..(t + 1) * hd];
        let h: Vec<f64> = (0..id)
            .map(|n| {
                let gv: f64 = (0..hd).map(|k| g[n * hd + k] * xt[k] as f64).sum();
                let uv: f64 = (0..hd).map(|k| u[n * hd + k] * xt[k] as f64).sum();
                uv * gv / (1.0 + (-gv).exp())
            })
            .collect();
        for n in 0..hd {
            y[t * hd + n] = (0..id).map(|k| d[n * id + k] * h[k]).sum::<f64>() as f32;
        }
    }
    Ok(y)
}

fn check_input(w: &ExpertWeights, x: &[f32], tokens: usize) -> Result<()> {
    if x.len() != tokens * w.hidden() {
        return Err(Error::Shape(format!("expert input has {} values for {tokens} x {}", x.len(), w.hidden())));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// executor
// ---------------------------------------------------------------------------

/// Per-job scheduling counters, guarded by the executor mutex.
#[derive(Default)]
struct Sched {
    c1: usize,
    c2: usize,
    n1_claim: usize,
    n1_done: usize,
    n2_claim: usize,
    n2_done: usize,
}

struct JobInner {
    w: ExpertWeights,
    x: Vec<f32>,
    tokens: usize,
    bf16: bool,
    /// `[T, I]`, written by stage-1 chunks (disjoint columns), read by
    /// stage 2. Only ever accessed through `h_ptr`.
    _h: Vec<f32>,
    h_ptr: *mut f32,
    /// `[T, H]`, written by stage-2 chunks (disjoint columns), taken by the
    /// owner once finished.
    y: UnsafeCell<Vec<f32>>,
    y_ptr: *mut f32,
    sched: UnsafeCell<Sched>,
    error: Mutex<Option<Error>>,
    finished: AtomicBool,
}

// SAFETY: h/y are only written through raw pointers to disjoint column
// ranges handed out under the executor mutex; stage 2 starts after every stage-1 chunk has
// reported done under that mutex; the owner reads y only after `finished`
// (release/acquire). `sched` is only touched with the mutex held.
unsafe impl Send for JobInner {}
unsafe impl Sync for JobInner {}

/// Handle to one submitted expert computation.
pub struct ExpertJob {
    inner: Arc<JobInner>,
}

impl ExpertJob {
    pub fn is_done(&self) -> bool {
        self.inner.finished.load(Ordering::Acquire)
    }

    /// Spin (with yields) until done, then take the `[T, H]` output.
    pub fn wait(self) -> Result<Vec<f32>> {
        while !self.is_done() {
            std::thread::yield_now();
        }
        self.take()
    }

    /// The `[T, H]` output; the job must be done.
    pub fn take(self) -> Result<Vec<f32>> {
        assert!(self.is_done(), "ExpertJob::take before completion");
        if let Some(e) = self.inner.error.lock().unwrap().take() {
            return Err(e);
        }
        // SAFETY: finished => every writer is gone; workers only touch
        // `sched` (under the mutex) after this point.
        Ok(std::mem::take(unsafe { &mut *self.inner.y.get() }))
    }
}

struct State {
    active: VecDeque<Arc<JobInner>>,
    pending: usize,
    stop: bool,
}

struct Shared {
    state: Mutex<State>,
    /// "work may be available"
    work: Condvar,
    /// "everything finished"
    idle: Condvar,
}

struct Work {
    job: Arc<JobInner>,
    stage: u8,
    idx: usize,
}

impl Shared {
    /// Caller holds the lock.
    fn find_work(&self, st: &mut State) -> Option<Work> {
        let mut i = 0;
        while i < st.active.len() {
            let job = st.active[i].clone();
            // SAFETY: lock held.
            let s = unsafe { &mut *job.sched.get() };
            if s.n1_claim < s.c1 {
                s.n1_claim += 1;
                return Some(Work { job, stage: 1, idx: s.n1_claim - 1 });
            }
            if s.n1_done == s.c1 && s.n2_claim < s.c2 {
                s.n2_claim += 1;
                if s.n2_claim == s.c2 {
                    st.active.remove(i);
                }
                return Some(Work { job, stage: 2, idx: s.n2_claim - 1 });
            }
            i += 1; // stage 1 still running elsewhere
        }
        None
    }

    fn run(w: &Work) -> Result<()> {
        let j = &w.job;
        let n0 = w.idx * CHUNK_ROWS;
        // SAFETY: chunk ranges are disjoint and stage ordering is enforced
        // by the scheduler (see JobInner).
        unsafe {
            if w.stage == 1 {
                gateup_rows(&j.w, &j.x, j.tokens, j.h_ptr, n0, (n0 + CHUNK_ROWS).min(j.w.intermediate()), j.bf16)
            } else {
                let h = std::slice::from_raw_parts(j.h_ptr, j.tokens * j.w.intermediate());
                down_rows(&j.w, h, j.tokens, j.y_ptr, n0, (n0 + CHUNK_ROWS).min(j.w.hidden()), j.bf16)
            }
        }
    }

    /// Caller holds the lock.
    fn mark_done(&self, st: &mut State, w: &Work) {
        // SAFETY: lock held.
        let s = unsafe { &mut *w.job.sched.get() };
        if w.stage == 1 {
            s.n1_done += 1;
            if s.n1_done == s.c1 {
                self.work.notify_all(); // stage 2 unlocked
            }
        } else {
            s.n2_done += 1;
            if s.n2_done == s.c2 {
                w.job.finished.store(true, Ordering::Release);
                st.pending -= 1;
                if st.pending == 0 {
                    self.idle.notify_all();
                }
            }
        }
    }

    fn worker_loop(&self) {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(w) = self.find_work(&mut st) {
                drop(st);
                if let Err(e) = Self::run(&w) {
                    w.job.error.lock().unwrap().get_or_insert(e);
                }
                st = self.state.lock().unwrap();
                self.mark_done(&mut st, &w);
                continue;
            }
            if st.stop {
                return;
            }
            st = self.work.wait(st).unwrap();
        }
    }
}

pub struct ExpertExecutor {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ExpertExecutor {
    /// `threads == 0` picks `available_parallelism - reserve` (at least 1).
    pub fn new(threads: usize, reserve: usize) -> Self {
        let threads = if threads > 0 {
            threads
        } else {
            let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
            hw.saturating_sub(reserve).max(1)
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(State { active: VecDeque::new(), pending: 0, stop: false }),
            work: Condvar::new(),
            idle: Condvar::new(),
        });
        let workers = (0..threads)
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("garlic-expert-{i}"))
                    .spawn(move || s.worker_loop())
                    .expect("spawn expert worker")
            })
            .collect();
        ExpertExecutor { shared, workers }
    }

    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// Queue one expert over `tokens` rows of `x` (`[tokens, H]`). Returns
    /// immediately.
    pub fn submit(&self, w: ExpertWeights, x: Vec<f32>, tokens: usize, bf16_rounding: bool) -> Result<ExpertJob> {
        check_input(&w, &x, tokens)?;
        let (hd, id) = (w.hidden(), w.intermediate());
        let sched = Sched { c1: id.div_ceil(CHUNK_ROWS), c2: hd.div_ceil(CHUNK_ROWS), ..Default::default() };
        let empty = tokens == 0;
        let mut h = vec![0f32; tokens * id];
        let mut y = vec![0f32; tokens * hd];
        let (h_ptr, y_ptr) = (h.as_mut_ptr(), y.as_mut_ptr());
        let inner = Arc::new(JobInner {
            w,
            x,
            tokens,
            bf16: bf16_rounding,
            _h: h,
            h_ptr,
            y: UnsafeCell::new(y),
            y_ptr,
            sched: UnsafeCell::new(sched),
            error: Mutex::new(None),
            finished: AtomicBool::new(empty),
        });
        if !empty {
            let mut st = self.shared.state.lock().unwrap();
            st.active.push_back(inner.clone());
            st.pending += 1;
            drop(st);
            self.shared.work.notify_all();
        }
        Ok(ExpertJob { inner })
    }

    /// Block until every submitted job has finished.
    pub fn wait_all(&self) {
        let mut st = self.shared.state.lock().unwrap();
        while st.pending > 0 {
            st = self.shared.idle.wait(st).unwrap();
        }
    }
}

impl Drop for ExpertExecutor {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Process-wide executor, sized by `GARLIC_CPU_THREADS` (default: all cores).
pub fn global() -> &'static ExpertExecutor {
    static EXEC: OnceLock<ExpertExecutor> = OnceLock::new();
    EXEC.get_or_init(|| {
        let threads = std::env::var("GARLIC_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        ExpertExecutor::new(threads, 0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    fn data(len: usize, seed: f32, amp: f32) -> Vec<f32> {
        (0..len).map(|i| ((i as f32 + seed) * 0.618).sin() * amp).collect()
    }

    fn fp8_weight(n: usize, k: usize, seed: f32) -> Tensor {
        let mut t = Tensor::from_f32(&data(n * k, seed, 1.0), &[n, k]).unwrap().to_dtype(DType::F8E4M3).unwrap();
        let (sn, sk) = (n.div_ceil(128), k.div_ceil(128));
        let s: Vec<f32> = (0..sn * sk).map(|i| 0.02 + 0.01 * i as f32).collect();
        t.set_scale(Tensor::from_f32(&s, &[sn, sk]).unwrap()).unwrap();
        t
    }

    fn expert(h: usize, i: usize) -> ExpertWeights {
        ExpertWeights::new(fp8_weight(i, h, 1.0), fp8_weight(i, h, 2.0), fp8_weight(h, i, 3.0)).unwrap()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{x} vs {y}");
        }
    }

    #[test]
    fn single_thread_matches_reference() {
        let (h, i) = (200, 136); // partial 128-blocks
        let w = expert(h, i);
        for t in [1, 3, 4, 6] {
            let x = data(t * h, 7.0, 0.5);
            close(&expert_ffn(&w, &x, t, false).unwrap(), &expert_ffn_ref(&w, &x, t).unwrap(), 1e-4);
        }
    }

    #[test]
    fn bf16_rounding_rounds_output() {
        let w = expert(128, 64);
        let x = data(2 * 128, 1.0, 0.5);
        let y = expert_ffn(&w, &x, 2, true).unwrap();
        assert!(y.iter().all(|&v| dtype::round_bf16(v) == v));
        close(&y, &expert_ffn_ref(&w, &x, 2).unwrap(), 2e-2);
    }

    #[test]
    fn executor_matches_single_thread_bitwise() {
        let exec = ExpertExecutor::new(3, 0);
        let (h, i) = (192, 320);
        let experts: Vec<ExpertWeights> = (0..5).map(|_| expert(h, i)).collect();
        let jobs: Vec<(ExpertJob, Vec<f32>)> = experts
            .iter()
            .enumerate()
            .map(|(e, w)| {
                let t = e + 1;
                let x = data(t * h, e as f32, 0.5);
                let want = expert_ffn(w, &x, t, false).unwrap();
                (exec.submit(w.clone(), x, t, false).unwrap(), want)
            })
            .collect();
        exec.wait_all();
        for (job, want) in jobs {
            assert!(job.is_done());
            assert_eq!(job.take().unwrap(), want);
        }
        // zero tokens completes immediately
        let empty = exec.submit(experts[0].clone(), vec![], 0, false).unwrap();
        assert!(empty.wait().unwrap().is_empty());
    }
}
