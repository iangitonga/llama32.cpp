#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#include "metrics.h"

#if defined(__AVX__)
#include <immintrin.h>

#define SIMD_AVX_LANES 8
#endif

#if defined(__NVCC__)
#define LL32_CUDA_N_BLOCKS 1
#define LL32_CUDA_N_THREADS 512
#endif


typedef uint16_t Float16;


enum class Device {
    CPU,
    CUDA
};

struct InferenceState {
    const int n_vocab = 128256;
    const int d_embd = 2048;
    const int n_heads = 32;
    const int n_kv_heads = 8;
    const int d_head = 64;
    const int d_mlp = 8192;
    const float rms_norm_eps = 1e-05f;
    const float qk_scaler = 1.0f / sqrtf(64); // => 1 / d_head
    Device device;
    int n_ctx = 0; // context size.
    // Start index of the context for uncached computations. I.e Computations for tokens at pos 0
    // to (start_pos - 1) in the context until pos are cached already.
    int start_pos = 0;

    InferenceState(Device device_)
        : device{device_}
    {}
};


namespace fpcvt {

// FP32 <-> FP16 Conversions.
#if defined(__NVCC__)
__host__ __device__
#endif
float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

#if defined(__NVCC__)
__host__ __device__
#endif
uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

#if defined(__NVCC__)
__host__ __device__
#endif
float fp16_to_fp32(Float16 h)
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

#if defined(__NVCC__)
__host__ __device__
#endif
inline Float16 fp32_to_fp16(float f) noexcept
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static float* init_fpcvt_cache() {
    // TODO: fix memory leak.
    float* cache = new float[65536];
    Float16 idx = 0;
    for (int i = 0; i < 65536; i++) {
        cache[i] = fp16_to_fp32(idx);
        idx += 1;
    }
    return cache;
}

// Global lookup table for fp16->fp32 to avoid recomputations.
static const float* G_fp16_to_fp32_table = init_fpcvt_cache();

} // namespace fpcvt


// Convert 16-bit float to 32-bit float.
[[nodiscard]]
inline float f16_to_f32_cpu(Float16 half) {
#if defined(__F16C__)
    return _cvtsh_ss(half);
#else 
    return fpcvt::G_fp16_to_fp32_table[half];
#endif
}

#if defined(__NVCC__)
[[nodiscard]]
__device__
inline float f16_to_f32_cuda(Float16 half) {
    return fpcvt::fp16_to_fp32(half);
}
#endif


// Convert 32-bit float to 16-bit float.
[[nodiscard]]
inline Float16 f32_to_f16_cpu(float flt) {
#if defined(__F16C__)
    return _cvtss_sh(flt, 0);
#else
    return fpcvt::fp32_to_fp16(flt);
#endif
}

#if defined(__NVCC__)
[[nodiscard]]
__device__
inline Float16 f32_to_f16_cuda(float flt) {
    return fpcvt::fp32_to_fp16(flt);
}
#endif


namespace ops {

int ceil_div(int n, int m) {
    return static_cast<int>(std::ceil(((float)n)/(float)m));
}

// tokens   : (n_ctx)
// emb_table: (n_vocab, d_embd)
// out      : (n_ctx, d_embd)
void embed_f16_cpu(const int* tokens, Float16* emb_table, Float16* out, int n_vocab, int n_ctx, int d_embd, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        const int emb_table_idx = tokens[i];
        const void* src = emb_table + emb_table_idx * d_embd;
        void* dest = out + i * d_embd;
        const size_t cpy_size = d_embd * sizeof(Float16);
        memcpy(dest, src, cpy_size);
    }
}

#if defined(__NVCC__)
__host__
void embed_f16_cuda(const int* tokens, Float16* emb_table, Float16* out, int n_vocab, int n_ctx, int d_embd, int start_pos) {
    for (int i = start_pos; i < n_ctx; i++) {
        const int emb_table_idx = tokens[i];
        const void* src = emb_table + emb_table_idx * d_embd;
        void* dest = out + i * d_embd;
        const size_t cpy_size = d_embd * sizeof(Float16);
        cudaMemcpy(dest, src, cpy_size, cudaMemcpyDeviceToDevice);
    }
}
#endif

void embed(const int* tokens, char* emb_table, char* out, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch(s.device) {
        case Device::CPU: {
            embed_f16_cpu(tokens, (Float16*)emb_table, (Float16*)out, s.n_vocab, s.n_ctx, s.d_embd, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            embed_f16_cuda(tokens, (Float16*)emb_table, (Float16*)out, s.n_vocab, s.n_ctx, s.d_embd, s.start_pos);
            cudaDeviceSynchronize();
#endif
            break;
        }
    }
}

// rms_norm(x_i) = x_i * 1/rms(x) * weight(i) where rms(x) = sqrt(1/n * sum(x*x))
// inp   : (n_ctx, d_embd)
// weight: (d_embd)
// out   : (n_ctx, d_embd)
void rms_norm_f16_cpu(const Float16* inp, const Float16* weight, Float16* out, int n_ctx, int d_embd, int start_pos, float eps)
{
    for (int i = start_pos; i < n_ctx; i++) {
        // compute mean of val squared.
        float sum_squares = 0.0f;
        for (int j = 0; j < d_embd; j++) {
            /// TODO: Use predefined pow fn.
            sum_squares += f16_to_f32_cpu(inp[i * d_embd + j]) * f16_to_f32_cpu(inp[i * d_embd + j]);
        }
        const float rms = sqrtf(sum_squares / (float)d_embd);
        const float rsqrt = 1.0f / (rms + eps);
        
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = f32_to_f16_cpu(f16_to_f32_cpu(inp[i * d_embd + j]) * rsqrt * f16_to_f32_cpu(weight[j]));
        }
        // x = xi / (root_mean_sq + 1e-6f) * wi
        // x = x / (rms+eps) * weight
    }
}

#if defined(__NVCC__)
__global__
void rms_norm_f16_cuda(const Float16* inp, const Float16* weight, Float16* out, int n_ctx, int d_embd, int start_pos, float eps)
{
    const int th_idx = threadIdx.x;
    const int th_stride = blockDim.x;

    for (int i = start_pos + th_idx; i < n_ctx; i += th_stride) {
        float sum_squares = 0.0f;
        for (int j = 0; j < d_embd; j++) {
            /// TODO: Use predefined pow fn.
            sum_squares += f16_to_f32_cuda(inp[i * d_embd + j]) * f16_to_f32_cuda(inp[i * d_embd + j]);
        }
        const float rms = sqrtf(sum_squares / (float)d_embd);
        const float rsqrt = 1.0f / (rms + eps);
        
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = f32_to_f16_cuda(f16_to_f32_cuda(inp[i * d_embd + j]) * rsqrt * f16_to_f32_cuda(weight[j]));
        }
    }
}
#endif


void rms_norm(const char* inp, const char* weight, char* out, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            rms_norm_f16_cpu((Float16*)inp, (Float16*)weight, (Float16*)out, s.n_ctx, s.d_embd, s.start_pos, s.rms_norm_eps);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            rms_norm_f16_cuda<<<LL32_CUDA_N_BLOCKS, LL32_CUDA_N_THREADS>>>((Float16*)inp, (Float16*)weight, (Float16*)out, s.n_ctx, s.d_embd, s.start_pos, s.rms_norm_eps);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}


// inp0: (n_ctx, d_embd)
// inp1: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void residual_f16_cpu(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_embd, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = f32_to_f16_cpu(f16_to_f32_cpu(inp0[i * d_embd + j]) + f16_to_f32_cpu(inp1[i * d_embd + j]));
        }
    }
}

#if defined(__NVCC__)
__global__
void residual_f16_cuda(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_embd, int start_pos)
{
    const int th_idx = threadIdx.x;
    const int th_stride = blockDim.x;

    for (int i = start_pos + th_idx; i < n_ctx; i += th_stride) {
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = f32_to_f16_cuda(f16_to_f32_cuda(inp0[i * d_embd + j]) + f16_to_f32_cuda(inp1[i * d_embd + j]));
        }
    }
}
#endif

void residual(const char* inp0, const char* inp1, char* out, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            residual_f16_cpu((Float16*)inp0, (Float16*)inp1, (Float16*)out, s.n_ctx, s.d_embd, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            residual_f16_cuda<<<LL32_CUDA_N_BLOCKS, LL32_CUDA_N_THREADS>>>((Float16*)inp0, (Float16*)inp1, (Float16*)out, s.n_ctx, s.d_embd, s.start_pos);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}

// inp0: (n_ctx, d_embd)
// inp1: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void mul_inplace_f16_cpu(Float16* inp0, const Float16* inp1, int n_ctx, int d_embd, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            inp0[i * d_embd + j] = f32_to_f16_cpu(f16_to_f32_cpu(inp0[i * d_embd + j]) * f16_to_f32_cpu(inp1[i * d_embd + j]));
        }
    }
}

#if defined(__NVCC__)
__global__
void mul_inplace_f16_cuda(Float16* inp0, const Float16* inp1, int n_ctx, int d_embd, int start_pos)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x + start_pos;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_ctx && j < d_embd) {
        inp0[i * d_embd + j] = f32_to_f16_cuda(f16_to_f32_cuda(inp0[i * d_embd + j]) * f16_to_f32_cuda(inp1[i * d_embd + j]));
    }
}
#endif

void mul_inplace(char* inp0, const char* inp1, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            mul_inplace_f16_cpu((Float16*)inp0, (Float16*)inp1, s.n_ctx, s.d_mlp, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            dim3 grid_dim(ceil_div(s.n_ctx-s.start_pos, 16), ceil_div(s.d_mlp, 16), 1);
            dim3 block_dim(16, 16, 1);
            mul_inplace_f16_cuda<<<grid_dim, block_dim>>>((Float16*)inp0, (Float16*)inp1, s.n_ctx, s.d_mlp, s.start_pos);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}

// inp: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void silu_inplace_f16_cpu(Float16* inp, int n_ctx, int d_embd, int start_pos)
{
     for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            const float x = f16_to_f32_cpu(inp[i * d_embd + j]);
            inp[i * d_embd + j] = f32_to_f16_cpu(x / (1.0f + expf(-x)));
        }
    }
}


#if defined(__NVCC__)
__global__
void silu_inplace_f16_cuda(Float16* inp, int n_ctx, int d_embd, int start_pos)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x + start_pos;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

     if (i < n_ctx && j < d_embd) {
          const float x = f16_to_f32_cuda(inp[i * d_embd + j]);
          inp[i * d_embd + j] = f32_to_f16_cuda(x / (1.0f + expf(-x)));
    }
}
#endif

void silu_inplace(char* inp, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            silu_inplace_f16_cpu((Float16*)inp, s.n_ctx, s.d_mlp, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            dim3 grid_dim(ceil_div(s.n_ctx-s.start_pos, 16), ceil_div(s.d_mlp, 16), 1);
            dim3 block_dim(16, 16, 1);
            silu_inplace_f16_cuda<<<grid_dim, block_dim>>>((Float16*)inp, s.n_ctx, s.d_mlp, s.start_pos);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}


#if defined(__AVX__)
__m256 vec_f32x8_load(const Float16* src_ptr) {
#if defined(__F16C__)
    return _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(const_cast<Float16*>(src_ptr))));
#else
    float f32[SIMD_AVX_LANES];
    for (int i = 0; i < SIMD_AVX_LANES; ++i) {
        f32[i] = f16_to_f32_cpu(src_ptr[i]);
    }
    return _mm256_loadu_ps(f32);
#endif
}

float avx_reduce_sum(const __m256 x)
{
    const __m128 hi_quad = _mm256_extractf128_ps(x, 1);
    const __m128 lo_quad = _mm256_castps256_ps128(x);
    const __m128 sum_quad = _mm_add_ps(lo_quad, hi_quad);
    const __m128 lo_dual = sum_quad;
    const __m128 hi_dual = _mm_movehl_ps(sum_quad, sum_quad);
    const __m128 sum_dual = _mm_add_ps(lo_dual, hi_dual);
    const __m128 lo = sum_dual;
    const __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
#endif


float vec_dot_product_f16_cpu(const Float16* vec_a, const Float16* vec_b, int vec_size)
{
#if defined(__AVX__)
    const int simd_vec_size = (int)(vec_size / SIMD_AVX_LANES) * SIMD_AVX_LANES;
    
    __m256 dot_prod_accum = _mm256_setzero_ps();
    for (int i = 0; i < simd_vec_size; i += SIMD_AVX_LANES) {
        const __m256 x0 = vec_f32x8_load(vec_a + i);
        const __m256 x1 = vec_f32x8_load(vec_b + i);
        dot_prod_accum = _mm256_add_ps(_mm256_mul_ps(x0, x1), dot_prod_accum);
    }
    
    // const float* f = (float *)(&dot_prod_accum);
    /// TODO:  Improve this: use simd to reduce sum.
    // float dot_prod = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
    float dot_prod = avx_reduce_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = f16_to_f32_cpu(vec_a[i]);
        const float x1 = f16_to_f32_cpu(vec_b[i]);
        dot_prod += x0 * x1;
    }

#else
    float dot_prod = 0.0f;

    for (int i = 0; i < vec_size; i += 1) {
        dot_prod += f16_to_f32_cpu(vec_a[i]) * f16_to_f32_cpu(vec_b[i]);
    }

#endif

    return dot_prod;
}

#if defined(__NVCC__)
__device__
void vec_dot_product_f16_cuda(const Float16* vec_a, const Float16* vec_b, int vec_size, float* out) {
    float dot_prod = 0.0f;
    for (int i = 0; i < vec_size; i += 1) {
        dot_prod += f16_to_f32_cuda(vec_a[i]) * f16_to_f32_cuda(vec_b[i]);
    }
    *out = dot_prod;
}
#endif

// Computes logits for next-token pred only.
// inp   : n_ctx, d_embd
// weight: n_vocab, d_embd
// out   : n_vocab 
void lm_head_proj_f16_cpu(const Float16* inp, const Float16* weight, float* out, int n_vocab, int n_ctx, int d_embd)
{
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2)
#endif
    for (int i = n_ctx - 1; i < n_ctx; i++) {
        for (int j = 0; j < n_vocab; j++) {
            const float dot_prod = vec_dot_product_f16_cpu(inp + i * d_embd, weight + j*d_embd, d_embd);
            out[j] = dot_prod;
        }
    }
}


#if defined(__NVCC__)
__global__
void lm_head_proj_f16_cuda(const Float16* inp, const Float16* weight, float* out, int n_vocab, int n_ctx, int d_embd)
{
    const int i = n_ctx - 1;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n_vocab) {
            float dot_prod;
            vec_dot_product_f16_cuda(inp + i * d_embd, weight + j*d_embd, d_embd, &dot_prod);
            out[j] = dot_prod;
    }
}
#endif

void lm_head_proj(const char* inp, const char* weight, float* out, const InferenceState& s)
{
    Timer timer{&metrics.matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            lm_head_proj_f16_cpu((Float16*)inp, (Float16*)weight, out, s.n_vocab, s.n_ctx, s.d_embd);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            dim3 grid_dim(ceil_div(s.n_vocab, 256), 1, 1);
            dim3 block_dim(16*16, 1, 1);
            lm_head_proj_f16_cuda<<<grid_dim, block_dim>>>((Float16*)inp, (Float16*)weight, out, s.n_vocab, s.n_ctx, s.d_embd);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}


// inp0: (n_ctx, d_in)
// inp1: (d_out, d_in)
// out : (n_ctx, d_out)
void matmul_2d_f16_cpu(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos)
{
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2)
#endif
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_out; j++) {
            const float dot_prod = vec_dot_product_f16_cpu(inp0 + i*d_in, inp1 + j*d_in, d_in);
            out[i * d_out + j] = f32_to_f16_cpu(dot_prod);
        }
    }   
}

#if defined(__NVCC__)
__global__
void matmul_2d_f16_cuda_naive_impl(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    i = i + start_pos;

    if (i < n_ctx && j < d_out) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_in; k += 1) {
                dot_prod += f16_to_f32_cuda(inp0[i*d_in + k]) * f16_to_f32_cuda(inp1[j*d_in + k]);
            }
            out[i * d_out + j] = f32_to_f16_cuda(dot_prod);
    }   
}
#endif

void matmul_2d_f16_cuda_naive(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos) {
#if defined(__NVCC__)
    int ctx_size = n_ctx - start_pos;
    dim3 block_dim(16, 16);
    int n_ctx_d = (int)std::ceil(((float)ctx_size)/16.0f);
    int d_out_d = (int)std::ceil(((float)d_out)/16.0f);
    dim3 grid_dim(n_ctx_d, d_out_d);
    matmul_2d_f16_cuda_naive_impl<<<grid_dim, block_dim>>>(inp0, inp1, out, n_ctx, d_in, d_out, start_pos);
    cudaDeviceSynchronize();
#endif
}

/*
Memory access for this kernel is coalesced such that threads in a single warp
access consecutive memory.

For execution, the threads of a block are grouped into so-called warps,
consisting of 32 threads. The grouping into warps happens based on a
consecutive threadId. Then, threads with neighbouring threadId become part of
the same warp.

In a GPU sequential memory accesses by threads that are part of the same warp
can be grouped and executed as one. This is referred to as global memory
coalescing. It’s the most important thing to keep in mind when optimizing a
kernel’s GMEM memory accesses toward achieving the peak bandwidth.

In this kernel, we utilise a 1-dim threadblock and then use indices such that
consecutive threads will access consecutive memory in inp0 and out matrices.

In the naive implementation memory access is as follows:

blockIdx.x=0, threadIdx.x=0: i=0,j=0
    - We access row 0 from inp0, row 0 from inp1 and row0col0 in out.
blockIdx.x=0, threadIdx.x=1: i=1,j=0
    - We access row 1 from inp0, row 0 from inp1 and row1col0 in out.
- Access for inp1 is consecutive but inp0 and out accesses are not consecutive.

In this implementation:

blockIdx.x=0, threadIdx.x=0: i=0,j=0
    - We access row 0 from inp0, row 0 from inp1 and row0col0 in out.
blockIdx.x=0, threadIdx.x=1: i=0,j=1
    - We access row 0 from inp0, row 0 from inp1 and row0col0 in out.
- We have consecutive accesses for both inp0 and out which allows for much more
  coalescing.
*/
#if defined(__NVCC__)
__global__
void matmul_2d_f16_cuda_coalesced_impl(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos)
{
    int block_size = 16;
    int i = blockIdx.x * (blockDim.x / block_size) + (threadIdx.x / block_size);
    int j = blockIdx.y * (blockDim.x / block_size) + (threadIdx.x % block_size);

    i = i + start_pos;

    if (i < n_ctx && j < d_out) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_in; k += 1) {
                dot_prod += f16_to_f32_cuda(inp0[i*d_in + k]) * f16_to_f32_cuda(inp1[j*d_in + k]);
            }
            out[i * d_out + j] = f32_to_f16_cuda(dot_prod);
    }   
}
#endif

void matmul_2d_f16_cuda_coalesced(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos)
{
#if defined(__NVCC__)
    const int ctx_size = n_ctx - start_pos;
    dim3 block_dim(16*16);
    const int n_ctx_d = (int)std::ceil(((float)ctx_size)/16.0f);
    const int d_out_d = (int)std::ceil(((float)d_out)/16.0f);
    dim3 grid_dim(n_ctx_d, d_out_d);
    matmul_2d_f16_cuda_coalesced_impl<<<grid_dim, block_dim>>>(inp0, inp1, out, n_ctx, d_in, d_out, start_pos);
    cudaDeviceSynchronize();
#endif
}


#if defined(__NVCC__)
__global__
void matmul2d_f16_cuda_smem_impl(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos) {
    constexpr int block_size = 256;
    const int n_in_blocks = d_in / block_size;

    const int i = start_pos;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    float dot_prod = 0.0f;
    for (int b = 0; b < n_in_blocks; b++) {

        __shared__ Float16 inp0_hot[block_size];
        inp0_hot[threadIdx.x] = inp0[i * d_in + b*block_size + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < block_size; k += 1) {
            dot_prod += f16_to_f32_cuda(inp0_hot[k]) * f16_to_f32_cuda(inp1[j*d_in + b*block_size + k]);
        }
        __syncthreads();
    }
    
    out[i * d_out + j] = f32_to_f16_cuda(dot_prod);
}
#endif

void matmul2d_f16_cuda_smem(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_in, int d_out, int start_pos)
{
#if defined(__NVCC__)
    const int block_size = 256;
    const int n_blocks = d_out / block_size;
    matmul2d_f16_cuda_smem_impl<<<n_blocks, block_size>>>(inp0, inp1, out, n_ctx, d_in, d_out, start_pos);
    cudaDeviceSynchronize();
#endif
}


void matmul_2d(const char* inp0, const char* inp1, char* out, int d_in, int d_out, const InferenceState& s)
{
    Timer timer{&metrics.matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            matmul_2d_f16_cpu((Float16*)inp0, (Float16*)inp1, (Float16*)out, s.n_ctx, d_in, d_out, s.start_pos);
            break;
        }
        case Device::CUDA: {
            matmul_2d_f16_cuda_coalesced((Float16*)inp0, (Float16*)inp1, (Float16*)out, s.n_ctx, d_in, d_out, s.start_pos);
            break;
        }
    }
}


// q: (n_ctx, qn_embd) - (n_ctx, q_heads, d_head)[phy] -> (q_heads, n_ctx, d_head)[virt]
// k: (n_ctx, kn_embd) - (n_ctx, k_heads, d_head)[phy] -> (k_heads, n_ctx, d_head)[virt]
// out: (q_heads, n_ctx, n_ctx)
void qk_f16_cpu(const Float16* q, const Float16* k, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, float scaler, int start_pos)
{
    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            // Compute the dot products which are not subsequently masked.
            const int end_non_masked = i + 1; 
            for (int j = 0; j < end_non_masked; j++) {
                const int hk = h / q_group_size;
                const float dot_prod = vec_dot_product_f16_cpu(q + h * d_head + i * q_heads*d_head, k + hk*d_head + j * k_heads*d_head, d_head);
                out[h * n_ctx * n_ctx + i * n_ctx + j] = f32_to_f16_cpu(dot_prod * scaler);
            }
        }
    }
}


#if defined(__NVCC__)
__global__
void qk_f16_cuda(const Float16* q, const Float16* k, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, float scaler, int start_pos)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + start_pos;

    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

    if (h < q_heads && i < n_ctx) {
        // Compute the dot products which are not subsequently masked.
        const int end_non_masked = i + 1; 
        for (int j = 0; j < end_non_masked; j++) {
            const int hk = h / q_group_size;
            float dot_prod;
            vec_dot_product_f16_cuda(q + h * d_head + i * q_heads*d_head, k + hk*d_head + j * k_heads*d_head, d_head, &dot_prod);
            out[h * n_ctx * n_ctx + i * n_ctx + j] = f32_to_f16_cuda(dot_prod * scaler);
        }
    }
}
#endif

void qk(const char* q, const char* k, char* out, const InferenceState& s)
{
    Timer timer{&metrics.matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            qk_f16_cpu((Float16*)q, (Float16*)k, (Float16*)out, s.n_ctx, s.n_heads, s.n_kv_heads, s.d_head, s.qk_scaler, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            dim3 grid_dim(ceil_div(s.n_heads, 16), ceil_div(s.n_ctx-s.start_pos, 16), 1);
            dim3 block_dim(16, 16, 1);
            qk_f16_cuda<<<grid_dim, block_dim>>>((Float16*)q, (Float16*)k, (Float16*)out, s.n_ctx, s.n_heads, s.n_kv_heads, s.d_head, s.qk_scaler, s.start_pos);
            cudaDeviceSynchronize();
#endif            
            break;
        }
    }
}

// inp: (n_heads, n_ctx, n_ctx)
void attn_mask_inplace_f16_cpu(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int i = 0; i < n_heads; i++) {
        for (int j = start_pos; j < n_ctx; j++) {
            const int start_ix = j + 1;
            for (int k = start_ix; k < n_ctx; k++) {
                inp[i * n_ctx * n_ctx + j * n_ctx + k] = f32_to_f16_cpu(-INFINITY);
            }
        }
    }
}

#if defined(__NVCC__)
__global__
void attn_mask_inplace_f16_cuda(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    const int th_idx = threadIdx.x;
    const int th_stride = blockDim.x;

    for (int i = th_idx; i < n_heads; i += th_stride) {
        for (int j = start_pos; j < n_ctx; j++) {
            const int start_ix = j + 1;
            for (int k = start_ix; k < n_ctx; k++) {
                inp[i * n_ctx * n_ctx + j * n_ctx + k] = f32_to_f16_cuda(-INFINITY);
            }
        }
    }
}
#endif

void attn_mask_inplace(char* inp, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            attn_mask_inplace_f16_cpu((Float16*)inp, s.n_heads, s.n_ctx, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            attn_mask_inplace_f16_cuda<<<LL32_CUDA_N_BLOCKS, LL32_CUDA_N_THREADS>>>((Float16*)inp, s.n_heads, s.n_ctx, s.start_pos);
            cudaDeviceSynchronize();
#endif
            break;
        }
    }
}

// inp: [n_heads, n_ctx, n_ctz]
void softmax_inplace_f16_cpu(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int h = 0; h < n_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            float max = -INFINITY;
            for (int j = 0; j < n_ctx; j++) {
                const float val = f16_to_f32_cpu(inp[h * n_ctx * n_ctx + i * n_ctx + j]);
                if (val > max) {
                    max = val;
                }
            }

            float sum_exp = 0;
            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                const float res = expf(f16_to_f32_cpu(inp[idx]) - max);
                sum_exp += res;
                inp[idx] = f32_to_f16_cpu(res);
            }

            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                inp[idx] = f32_to_f16_cpu(f16_to_f32_cpu(inp[idx]) / sum_exp);
            }
        }
    }
}

#if defined(__NVCC__)
__global__
void softmax_inplace_f16_cuda(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    const int th_idx = threadIdx.x;
    const int th_stride = blockDim.x;

    for (int h = th_idx; h < n_heads; h += th_stride) {
        for (int i = start_pos; i < n_ctx; i++) {
            float max = -INFINITY;
            for (int j = 0; j < n_ctx; j++) {
                const float val = f16_to_f32_cuda(inp[h * n_ctx * n_ctx + i * n_ctx + j]);
                if (val > max) {
                    max = val;
                }
            }

            float sum_exp = 0;
            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                const float res = expf(f16_to_f32_cuda(inp[idx]) - max);
                sum_exp += res;
                inp[idx] = f32_to_f16_cuda(res);
            }

            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                inp[idx] = f32_to_f16_cuda(f16_to_f32_cuda(inp[idx]) / sum_exp);
            }
        }
    }
}
#endif

void softmax_inplace(char* inp, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            softmax_inplace_f16_cpu((Float16*)inp, s.n_heads, s.n_ctx, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            softmax_inplace_f16_cuda<<<LL32_CUDA_N_BLOCKS, LL32_CUDA_N_THREADS>>>((Float16*)inp, s.n_heads, s.n_ctx, s.start_pos);
            cudaDeviceSynchronize();
#endif
            break;
        }
    }
}

// qk: (n_heads, n_ctx, n_ctx)
//  v: (n_ctx, vn_embd) - (n_ctx, v_heads, d_heads)[phy] - (v_heads, d_heads, n_ctx)[virt]
// out: (n_ctx, q_heads, d_head)
void qkv_f16_cpu(const Float16* qk, const Float16* v, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos)
{
    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif
    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            for (int j = 0; j < d_head; j++) {
                float dot_prod = 0.0f;
                for (int k = 0; k < n_ctx; k++) {
                    // index of the current head in v.
                    const int hv = h / qk_group_size;
                    dot_prod += f16_to_f32_cpu(qk[h * n_ctx*n_ctx + i * n_ctx + k]) * f16_to_f32_cpu(v[hv * d_head + j + k * v_heads*d_head]);
                }
                out[i * q_heads*d_head + h*d_head + j] = f32_to_f16_cpu(dot_prod);
            } 
        }
    }
}

#if defined(__NVCC__)
__global__
void qkv_f16_cuda(const Float16* qk, const Float16* v, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + start_pos;
    const int j = blockIdx.z * blockDim.z + threadIdx.z;

    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);
    
    if (h < q_heads && i < n_ctx && j < d_head) {
        float dot_prod = 0.0f;
        for (int k = 0; k < n_ctx; k++) {
            // index of the current head in v.
            const int hv = h / qk_group_size;
            dot_prod += f16_to_f32_cuda(qk[h * n_ctx*n_ctx + i * n_ctx + k]) * f16_to_f32_cuda(v[hv * d_head + j + k * v_heads*d_head]);
        }
        out[i * q_heads*d_head + h*d_head + j] = f32_to_f16_cuda(dot_prod);
    } 
        
}
#endif

void qkv(const char* qk, const char* v, char* out, const InferenceState& s)
{
    Timer timer{&metrics.matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            qkv_f16_cpu((Float16*)qk, (Float16*)v, (Float16*)out, s.n_ctx, s.n_heads, s.n_kv_heads, s.d_head, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            dim3 grid_dim(ceil_div(s.n_heads, 8), ceil_div(s.n_ctx-s.start_pos, 8), ceil_div(s.d_head, 8));
            dim3 block_dim(8, 8, 8);
            qkv_f16_cuda<<<grid_dim, block_dim>>>((Float16*)qk, (Float16*)v, (Float16*)out, s.n_ctx, s.n_heads, s.n_kv_heads, s.d_head, s.start_pos);
            cudaDeviceSynchronize();
#endif
            break;
        }
    }
}

// inp: [n_ctx, n_head, d_head]
void rotary_emb_f16_cpu(Float16* inp, int n_ctx, int n_heads, int d_head, int start_pos)
{
    for (int i = start_pos; i < n_ctx; ++i) {
       for (int h = 0; h < n_heads; ++h) {
            Float16* inp_vec = inp + i*n_heads*d_head + h*d_head;

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j) {
                const float x0 = f16_to_f32_cpu(inp_vec[j]);
                const float x1 = f16_to_f32_cpu(inp_vec[j + d_half]);
                
                const float d = (float)(d_head);
                const float base_theta = 500000.0f;

                float inv_freq = powf(base_theta, -(2.0f*j/d));

                { // llama 3 rope modifications
                    const float factor = 32.0f;
                    const float low_freq_factor = 1.0f;
                    const float high_freq_factor = 4.0f;
                    const float old_context_len = 8192.0f;

                    const float low_freq_wavelen = old_context_len / low_freq_factor;
                    const float high_freq_wavelen = old_context_len / high_freq_factor;

                    const float pi = 3.141592653589793f;
                    const float wavelen = 2 * pi / inv_freq;

                    float inv_freq_llama = inv_freq;

                    if (wavelen > low_freq_wavelen) {
                        inv_freq_llama = inv_freq / factor;
                    }
                    if ((wavelen <= low_freq_wavelen) && (wavelen >= high_freq_wavelen)) {
                        const float smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        const float smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama;
                        inv_freq_llama = smoothed_inv_freq;
                    }

                    inv_freq = inv_freq_llama;
                }

                const float m = (float)(i);
                const float m_theta_i = m * inv_freq;

                const float o0 = x0 * cosf(m_theta_i) - x1 * sinf(m_theta_i);
                const float o1 = x0 * sinf(m_theta_i) + x1 * cosf(m_theta_i);

                inp_vec[j] = f32_to_f16_cpu(o0);
                inp_vec[j + d_half] = f32_to_f16_cpu(o1);
            }
        }
    }
}

#if defined(__NVCC__)
__global__
void rotary_emb_f16_cuda(Float16* inp, int n_ctx, int n_heads, int d_head, int start_pos)
{
    const int th_idx = threadIdx.x;
    const int th_stride = blockDim.x;

    for (int i = start_pos; i < n_ctx; ++i) {
       for (int h = th_idx; h < n_heads; h += th_stride) {
            Float16* inp_vec = inp + i*n_heads*d_head + h*d_head;

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j) {
                const float x0 = f16_to_f32_cuda(inp_vec[j]);
                const float x1 = f16_to_f32_cuda(inp_vec[j + d_half]);
                
                const float d = (float)(d_head);
                const float base_theta = 500000.0f;

                float inv_freq = powf(base_theta, -(2.0f*j/d));

                { // llama 3 rope modifications
                    const float factor = 32.0f;
                    const float low_freq_factor = 1.0f;
                    const float high_freq_factor = 4.0f;
                    const float old_context_len = 8192.0f;

                    const float low_freq_wavelen = old_context_len / low_freq_factor;
                    const float high_freq_wavelen = old_context_len / high_freq_factor;

                    const float pi = 3.141592653589793f;
                    const float wavelen = 2 * pi / inv_freq;

                    float inv_freq_llama = inv_freq;

                    if (wavelen > low_freq_wavelen) {
                        inv_freq_llama = inv_freq / factor;
                    }
                    if ((wavelen <= low_freq_wavelen) && (wavelen >= high_freq_wavelen)) {
                        const float smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                        const float smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama;
                        inv_freq_llama = smoothed_inv_freq;
                    }

                    inv_freq = inv_freq_llama;
                }

                const float m = (float)(i);
                const float m_theta_i = m * inv_freq;

                const float o0 = x0 * cosf(m_theta_i) - x1 * sinf(m_theta_i);
                const float o1 = x0 * sinf(m_theta_i) + x1 * cosf(m_theta_i);

                inp_vec[j] = f32_to_f16_cuda(o0);
                inp_vec[j + d_half] = f32_to_f16_cuda(o1);
            }
        }
    }
}
#endif


void rotary_emb(char* inp, int n_heads, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (s.device) {
        case Device::CPU: {
            rotary_emb_f16_cpu((Float16*)inp, s.n_ctx, n_heads, s.d_head, s.start_pos);
            break;
        }
        case Device::CUDA: {
#if defined(__NVCC__)
            rotary_emb_f16_cuda<<<LL32_CUDA_N_BLOCKS, LL32_CUDA_N_THREADS>>>((Float16*)inp, s.n_ctx, n_heads, s.d_head, s.start_pos);
            cudaDeviceSynchronize();
#endif
            break;
        }
    }
}


void copy_tensors(const char* src, char* dest, int n_ctx, int d_embd, const InferenceState& s)
{
    Timer timer{&metrics.non_matmul_ms};

    if (s.device == Device::CPU) {
        for (int i = s.start_pos; i < n_ctx; i++) {
            memcpy(dest + i * d_embd * sizeof(Float16), src + i * d_embd * sizeof(Float16), d_embd*sizeof(Float16));
        }
    }
    else {
#if defined(__NVCC__)
    for (int i = s.start_pos; i < n_ctx; i++) {
        cudaMemcpy(dest + i * d_embd * sizeof(Float16), src + i * d_embd * sizeof(Float16), d_embd*sizeof(Float16), cudaMemcpyDeviceToDevice);
    }
#endif
    }
}

void copy_cuda_to_host(const void* src, void* dest, size_t size) {
#if defined(__NVCC__)
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
#endif
}

void copy_host_to_cuda(const void* src, void* dest, size_t size) {
#if defined(__NVCC__)
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
#endif
}

} // namespace ops.
