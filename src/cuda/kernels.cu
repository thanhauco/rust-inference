// CUDA kernels for the garlic CUDA backend, compiled at runtime with NVRTC.
//
// No CUDA headers are included so the source compiles with a bare NVRTC
// install: bf16 / f16 values are carried as raw 16-bit patterns and converted
// with integer math (bf16) or PTX cvt instructions (f16). Activations are
// bf16 or f16 ("T" below); norm gains, rope tables, router weights and FP8
// block scales are f32. Every kernel is instantiated once per activation type
// as `<name>_bf16` / `<name>_f16`.

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define WARP 32

struct bf16 { u16 x; };
struct f16 { u16 x; };

__device__ __forceinline__ float to_f(bf16 v) { return __uint_as_float(((u32)v.x) << 16); }
__device__ __forceinline__ float to_f(f16 v) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v.x));
    return f;
}

template <typename T> __device__ __forceinline__ T from_f(float f);
template <> __device__ __forceinline__ bf16 from_f<bf16>(float f) {
    u32 u = __float_as_uint(f);
    bf16 r;
    if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN stays a quiet NaN
        r.x = (u16)((u >> 16) | 0x40u);
    } else {  // round to nearest even
        r.x = (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return r;
}
template <> __device__ __forceinline__ f16 from_f<f16>(float f) {
    f16 r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r.x) : "f"(f));
    return r;
}

// E4M3FN -> f32. Normal values get their exponent rebiased with an integer
// add (e + 120); subnormals are m * 2^-9. No float op ever sees a denormal.
__device__ __forceinline__ float fp8_to_f(u8 b) {
    u32 w = b;
    u32 sign = (w & 0x80u) << 24;
    if ((w & 0x78u) == 0) {
        return __uint_as_float(__float_as_uint((float)(w & 7u) * (1.0f / 512.0f)) | sign);
    }
    return __uint_as_float((((w & 0x7fu) << 20) + (120u << 23)) | sign);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, <= 1024).
__device__ float block_sum(float v) {
    __shared__ float part[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = (threadIdx.x < nw) ? part[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) part[0] = v;
    __syncthreads();
    float r = part[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ---------------------------------------------------------------------------
// elementwise / norms / rope
// ---------------------------------------------------------------------------

// out[t] = table[ids[t]]                     grid: tokens, block: 256
template <typename T>
__device__ void embedding(const u32* ids, const T* table, T* out, int dim) {
    const T* src = table + (long long)ids[blockIdx.x] * dim;
    T* dst = out + (long long)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// in-place RMSNorm over rows of `dim`          grid: rows, block: 256
template <typename T>
__device__ void rmsnorm(T* x, const float* w, int dim, float eps) {
    T* row = x + (long long)blockIdx.x * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = to_f(row[i]);
        ss += v * v;
    }
    float inv = rsqrtf(block_sum(ss) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) row[i] = from_f<T>(to_f(row[i]) * inv * w[i]);
}

// residual += x; x = rmsnorm(residual) * w    grid: rows, block: 256
template <typename T>
__device__ void add_rmsnorm(T* x, T* residual, const float* w, int dim, float eps) {
    T* xr = x + (long long)blockIdx.x * dim;
    T* rr = residual + (long long)blockIdx.x * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        T s = from_f<T>(to_f(xr[i]) + to_f(rr[i]));
        rr[i] = s;
        float v = to_f(s);
        ss += v * v;
    }
    float inv = rsqrtf(block_sum(ss) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) xr[i] = from_f<T>(to_f(rr[i]) * inv * w[i]);
}

// rotate-half RoPE on x: [tokens, heads, dh], cos/sin: [tokens, dh]
template <typename T>
__device__ void rope(T* x, const float* cos, const float* sin, int tokens, int heads, int dh) {
    int half = dh / 2;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)tokens * heads * half;
    if (i >= total) return;
    int d = (int)(i % half);
    long long th = i / half;  // token * heads + head
    int tok = (int)(th / heads);
    T* row = x + th * dh;
    float x1 = to_f(row[d]), x2 = to_f(row[d + half]);
    const float* c = cos + (long long)tok * dh;
    const float* s = sin + (long long)tok * dh;
    row[d] = from_f<T>(x1 * c[d] - x2 * s[d]);
    row[d + half] = from_f<T>(x2 * c[d + half] + x1 * s[d + half]);
}

template <typename T>
__device__ void swiglu(T* gate, const T* up, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gate[i] = from_f<T>(silu(to_f(gate[i])) * to_f(up[i]));
}

template <typename T>
__device__ void add(T* x, const T* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = from_f<T>(to_f(x[i]) + to_f(y[i]));
}

template <typename T>
__device__ void to_f32(const T* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_f(x[i]);
}

// ---------------------------------------------------------------------------
// reductions / routing
// ---------------------------------------------------------------------------

// first index of the row maximum               grid: rows, block: 256
template <typename T>
__device__ void argmax(const T* x, u32* out, int n) {
    __shared__ float sv[32];
    __shared__ u32 si[32];
    const T* row = x + (long long)blockIdx.x * n;
    float best = -3.402823466e38f;
    u32 bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = to_f(row[i]);
        if (v > best) { best = v; bi = i; }
    }
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        u32 oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = best; si[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) {
            if (sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; }
        }
        out[blockIdx.x] = bi;
    }
}

// MoE router: top-k of each row of logits [T, E] with softmax weights.
// normalize != 0: softmax over the selected logits (norm_topk_prob);
// otherwise full-softmax probabilities.     grid: T, block: 32, k <= 32
template <typename T>
__device__ void topk_softmax(const T* logits, u32* ids, float* weights, int e, int k, int normalize) {
    __shared__ u32 win[32];
    const T* row = logits + (long long)blockIdx.x * e;
    int lane = threadIdx.x;
    for (int j = 0; j < k; j++) {
        float best = -3.402823466e38f;
        u32 bi = 0xffffffffu;
        for (int i = lane; i < e; i += WARP) {
            bool taken = false;
            for (int m = 0; m < j; m++) taken |= (win[m] == (u32)i);
            float v = to_f(row[i]);
            if (!taken && (v > best || bi == 0xffffffffu)) { best = v; bi = i; }
        }
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            u32 oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > best || (ov == best && oi < bi))) { best = ov; bi = oi; }
        }
        if (lane == 0) win[j] = bi;
        __syncwarp();
    }
    float mx = to_f(row[win[0]]);
    float denom = 0.0f;
    if (normalize) {
        for (int j = 0; j < k; j++) denom += __expf(to_f(row[win[j]]) - mx);
    } else {
        float part = 0.0f;
        for (int i = lane; i < e; i += WARP) part += __expf(to_f(row[i]) - mx);
        denom = warp_sum(part);
    }
    if (lane < k) {
        ids[(long long)blockIdx.x * k + lane] = win[lane];
        weights[(long long)blockIdx.x * k + lane] = __expf(to_f(row[win[lane]]) - mx) / denom;
    }
}

// gathered[s] = x[token_of_slot[s]]            grid: slots, block: 256
template <typename T>
__device__ void moe_gather(const T* x, const u32* token_of_slot, T* out, int dim) {
    const T* src = x + (long long)token_of_slot[blockIdx.x] * dim;
    T* dst = out + (long long)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// out[t] = sum_j w[t, j] * expert_out[slot_of[t, j]]   grid: T, block: 256
template <typename T>
__device__ void moe_scatter(const T* expert_out, const float* w, const u32* slot_of, T* out, int dim, int k) {
    const float* wt = w + (long long)blockIdx.x * k;
    const u32* st = slot_of + (long long)blockIdx.x * k;
    T* dst = out + (long long)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < k; j++) acc += wt[j] * to_f(expert_out[(long long)st[j] * dim + i]);
        dst[i] = from_f<T>(acc);
    }
}

// ---------------------------------------------------------------------------
// FP8 block-scaled GEMV: y[m, n] = sum_k W[n, k] x[m, k]
//
// One warp per output row (8 rows per block), grid.y over activation rows.
// Partial sums are kept per k-group of `bk` columns and scaled once per
// group, like the C++ fp8_gemv kernels. Requires K % 4 == 0.
// ---------------------------------------------------------------------------

#define GEMV_ROWS 8

template <typename T>
__device__ __forceinline__ float fp8_row_dot(const u8* wrow, const float* srow, const T* xrow, int K, int bk, int lane) {
    float acc = 0.0f;
    for (int g0 = 0, g = 0; g0 < K; g0 += bk, g++) {
        int g1 = min(g0 + bk, K);
        float part = 0.0f;
        for (int k = g0 + lane * 4; k < g1; k += WARP * 4) {
            u32 packed = *reinterpret_cast<const u32*>(wrow + k);
            part += fp8_to_f((u8)(packed)) * to_f(xrow[k]);
            part += fp8_to_f((u8)(packed >> 8)) * to_f(xrow[k + 1]);
            part += fp8_to_f((u8)(packed >> 16)) * to_f(xrow[k + 2]);
            part += fp8_to_f((u8)(packed >> 24)) * to_f(xrow[k + 3]);
        }
        acc += part * srow[g];
    }
    return warp_sum(acc);
}

// grid: (ceil(N / 8), M), block: (32, 8)
template <typename T>
__device__ void fp8_gemv(T* y, const u8* w, const float* scales, const T* x, int N, int K, int M, int bn, int bk,
                         int scale_cols) {
    int n = blockIdx.x * GEMV_ROWS + threadIdx.y;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float acc = fp8_row_dot(w + (long long)n * K, scales + (long long)(n / bn) * scale_cols, x + (long long)m * K, K, bk,
                            threadIdx.x);
    if (threadIdx.x == 0) y[(long long)m * N + n] = from_f<T>(acc);
}

// Fused expert front half: y[m, n] = bf16(up[n].x) * bf16(silu(gate[n].x)),
// rounding exactly like the C++ grouped kernel.
template <typename T>
__device__ void fp8_gemv_swiglu(T* y, const u8* wg, const float* sg, const u8* wu, const float* su, const T* x, int N,
                                int K, int M, int bn, int bk, int scale_cols) {
    int n = blockIdx.x * GEMV_ROWS + threadIdx.y;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const T* xr = x + (long long)m * K;
    long long srow = (long long)(n / bn) * scale_cols;
    float g = fp8_row_dot(wg + (long long)n * K, sg + srow, xr, K, bk, threadIdx.x);
    float u = fp8_row_dot(wu + (long long)n * K, su + srow, xr, K, bk, threadIdx.x);
    if (threadIdx.x == 0) y[(long long)m * N + n] = from_f<T>(to_f(from_f<T>(u)) * to_f(from_f<T>(silu(g))));
}

// ---------------------------------------------------------------------------
// paged KV cache + attention
//
// k/v pages: [pages, kv_heads, 32, dh]; page_table: [B, max_pages_per_seq];
// lens_before[b]: tokens in sequence b before this step's append.
// ---------------------------------------------------------------------------

#define PAGE 32

// grid: (B * S, kv_heads), block: 128
template <typename T>
__device__ void kv_append(const T* k, const T* v, T* kp, T* vp, const int* page_table, const int* lens_before, int S,
                          int kv_heads, int dh, int max_pages_per_seq) {
    int bs = blockIdx.x, h = blockIdx.y;
    int b = bs / S, t = bs % S;
    int pos = lens_before[b] + t;
    int page = page_table[b * max_pages_per_seq + pos / PAGE];
    const T* ks = k + ((long long)bs * kv_heads + h) * dh;
    const T* vs = v + ((long long)bs * kv_heads + h) * dh;
    long long dst = (((long long)page * kv_heads + h) * PAGE + pos % PAGE) * dh;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) {
        kp[dst + i] = ks[i];
        vp[dst + i] = vs[i];
    }
}

#define ATTN_WARPS 4
#define MAX_DH 256

// Causal attention of one query row against its sequence's pages, online
// softmax. Each warp walks every ATTN_WARPS-th key; lane l owns head dims
// l, l+32, ... (< dh); warps are merged through shared memory at the end.
// q/out: [B, S, H, dh].   grid: (H, B * S), block: 32 * ATTN_WARPS
template <typename T>
__device__ void paged_attention(const T* q, const T* kp, const T* vp, T* out, const int* page_table,
                                const int* lens_before, int S, int heads, int kv_heads, int dh, int max_pages_per_seq,
                                float scale) {
    __shared__ float qs[MAX_DH];
    __shared__ float wm[ATTN_WARPS], wl[ATTN_WARPS];
    __shared__ float wacc[ATTN_WARPS][MAX_DH];

    int h = blockIdx.x, bs = blockIdx.y;
    int b = bs / S, t = bs % S;
    int kvh = h / (heads / kv_heads);
    int qpos = lens_before[b] + t;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const T* qrow = q + ((long long)bs * heads + h) * dh;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) qs[i] = to_f(qrow[i]);
    __syncthreads();

    const int per_lane = (dh + WARP - 1) / WARP;  // dh <= MAX_DH
    float acc[MAX_DH / WARP];
#pragma unroll
    for (int i = 0; i < MAX_DH / WARP; i++) acc[i] = 0.0f;
    float m = -3.402823466e38f, l = 0.0f;

    for (int j = warp; j <= qpos; j += ATTN_WARPS) {
        int page = page_table[b * max_pages_per_seq + j / PAGE];
        long long off = (((long long)page * kv_heads + kvh) * PAGE + j % PAGE) * dh;
        float s = 0.0f;
        for (int i = 0; i < per_lane; i++) {
            int d = lane + i * WARP;
            if (d < dh) s += qs[d] * to_f(kp[off + d]);
        }
        s = warp_sum(s) * scale;
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * corr + p;
        for (int i = 0; i < per_lane; i++) {
            int d = lane + i * WARP;
            if (d < dh) acc[i] = acc[i] * corr + p * to_f(vp[off + d]);
        }
        m = m_new;
    }

    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int i = 0; i < per_lane; i++) {
        int d = lane + i * WARP;
        if (d < dh) wacc[warp][d] = acc[i];
    }
    __syncthreads();

    float mx = wm[0];
    for (int w = 1; w < ATTN_WARPS; w++) mx = fmaxf(mx, wm[w]);
    float denom = 0.0f;
    for (int w = 0; w < ATTN_WARPS; w++) denom += wl[w] * __expf(wm[w] - mx);
    T* orow = out + ((long long)bs * heads + h) * dh;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) {
        float v = 0.0f;
        for (int w = 0; w < ATTN_WARPS; w++) v += wacc[w][i] * __expf(wm[w] - mx);
        orow[i] = from_f<T>(v / denom);
    }
}

// ---------------------------------------------------------------------------
// extern "C" entry points, one per activation type
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                                                               \
    extern "C" __global__ void embedding_##T(const u32* ids, const T* table, T* out, int dim) {                       \
        embedding<T>(ids, table, out, dim);                                                                          \
    }                                                                                                                \
    extern "C" __global__ void rmsnorm_##T(T* x, const float* w, int dim, float eps) { rmsnorm<T>(x, w, dim, eps); } \
    extern "C" __global__ void add_rmsnorm_##T(T* x, T* r, const float* w, int dim, float eps) {                     \
        add_rmsnorm<T>(x, r, w, dim, eps);                                                                           \
    }                                                                                                                \
    extern "C" __global__ void rope_##T(T* x, const float* c, const float* s, int tokens, int heads, int dh) {        \
        rope<T>(x, c, s, tokens, heads, dh);                                                                         \
    }                                                                                                                \
    extern "C" __global__ void swiglu_##T(T* g, const T* u, long long n) { swiglu<T>(g, u, n); }                     \
    extern "C" __global__ void add_##T(T* x, const T* y, long long n) { add<T>(x, y, n); }                           \
    extern "C" __global__ void to_f32_##T(const T* x, float* y, long long n) { to_f32<T>(x, y, n); }                 \
    extern "C" __global__ void argmax_##T(const T* x, u32* out, int n) { argmax<T>(x, out, n); }                     \
    extern "C" __global__ void topk_softmax_##T(const T* l, u32* ids, float* w, int e, int k, int norm) {            \
        topk_softmax<T>(l, ids, w, e, k, norm);                                                                      \
    }                                                                                                                \
    extern "C" __global__ void moe_gather_##T(const T* x, const u32* tos, T* out, int dim) {                         \
        moe_gather<T>(x, tos, out, dim);                                                                             \
    }                                                                                                                \
    extern "C" __global__ void moe_scatter_##T(const T* eo, const float* w, const u32* so, T* out, int dim, int k) {  \
        moe_scatter<T>(eo, w, so, out, dim, k);                                                                      \
    }                                                                                                                \
    extern "C" __global__ void fp8_gemv_##T(T* y, const u8* w, const float* s, const T* x, int N, int K, int M,       \
                                            int bn, int bk, int sc) {                                                \
        fp8_gemv<T>(y, w, s, x, N, K, M, bn, bk, sc);                                                                \
    }                                                                                                                \
    extern "C" __global__ void fp8_gemv_swiglu_##T(T* y, const u8* wg, const float* sg, const u8* wu,                 \
                                                   const float* su, const T* x, int N, int K, int M, int bn, int bk,  \
                                                   int sc) {                                                         \
        fp8_gemv_swiglu<T>(y, wg, sg, wu, su, x, N, K, M, bn, bk, sc);                                               \
    }                                                                                                                \
    extern "C" __global__ void kv_append_##T(const T* k, const T* v, T* kp, T* vp, const int* pt, const int* lens,   \
                                             int S, int kvh, int dh, int mpps) {                                     \
        kv_append<T>(k, v, kp, vp, pt, lens, S, kvh, dh, mpps);                                                      \
    }                                                                                                                \
    extern "C" __global__ void paged_attention_##T(const T* q, const T* kp, const T* vp, T* out, const int* pt,       \
                                                   const int* lens, int S, int heads, int kvh, int dh, int mpps,     \
                                                   float scale) {                                                    \
        paged_attention<T>(q, kp, vp, out, pt, lens, S, heads, kvh, dh, mpps, scale);                                \
    }

INSTANTIATE(bf16)
INSTANTIATE(f16)
