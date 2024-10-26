#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#include "metrics.h"


typedef uint16_t Float16;

enum class Dtype {
    Float16,
    Float32,
};

namespace fpcvt {

// FP32 <-> FP16 Conversions.
inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

inline float fp16_to_fp32(Float16 h) noexcept
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
inline float fp16_to_fp32(Float16 half) {
    return fpcvt::G_fp16_to_fp32_table[half];
}

// Convert 32-bit float to 16-bit float.
[[nodiscard]]
inline Float16 fp32_to_fp16(float flt) {
    return fpcvt::fp32_to_fp16(flt);
}

namespace ops {

// tokens   : (n_ctx)
// emb_table: (n_vocab, d_embd)
// out      : (n_ctx, d_embd)
void embed_f32(const int* tokens, float* emb_table, float* out, int n_vocab, int n_ctx, int d_embd)
{
    for (int i = 0; i < n_ctx; i++) {
        const int emb_table_idx = tokens[i];
        const void* src = emb_table + emb_table_idx * d_embd;
        void* dest = out + i * d_embd;
        const size_t cpy_size = d_embd * sizeof(float);
        std::memcpy(dest, src, cpy_size);
    }
}

void embed_f16(const int* tokens, Float16* emb_table, Float16* out, int n_vocab, int n_ctx, int d_embd)
{
    for (int i = 0; i < n_ctx; i++) {
        const int emb_table_idx = tokens[i];
        const void* src = emb_table + emb_table_idx * d_embd;
        void* dest = out + i * d_embd;
        const size_t cpy_size = d_embd * sizeof(Float16);
        std::memcpy(dest, src, cpy_size);
    }
}


void embed(const int* tokens, char* emb_table, char* out, int n_vocab, int n_ctx, int d_embd, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            embed_f16(tokens, (Float16*)emb_table, (Float16*)out, n_vocab, n_ctx, d_embd);
            break;
        }
        case Dtype::Float32: {
            embed_f32(tokens, (float*)emb_table, (float*)out, n_vocab, n_ctx, d_embd);
            break;
        }
    }
}

// rms_norm(x_i) = x_i * 1/rms(x) * weight(i) where rms(x) = sqrt(1/n * sum(x*x))
// inp   : (n_ctx, d_embd)
// weight: (d_embd)
// out   : (n_ctx, d_embd)
void rms_norm_f32(const float* inp, const float* weight, float* out, int n_ctx, int d_embd, float eps)
{
    for (int i = 0; i < n_ctx; i++) {
        // compute mean of val squared.
        float sum_squares = 0.0f;
        for (int j = 0; j < d_embd; j++) {
            /// TODO: Use predefined pow fn.
            sum_squares += inp[i * d_embd + j] * inp[i * d_embd + j];
        }
        const float rms = sqrtf(sum_squares / (float)d_embd);
        const float rsqrt = 1.0f / (rms + eps);
        
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] =  inp[i * d_embd + j] * rsqrt * weight[j];
        }
        // x = xi / (root_mean_sq + 1e-6f) * wi
        // x = x / (rms+eps) * weight
    }
}

void rms_norm_f16(const Float16* inp, const Float16* weight, Float16* out, int n_ctx, int d_embd, float eps)
{
    for (int i = 0; i < n_ctx; i++) {
        // compute mean of val squared.
        float sum_squares = 0.0f;
        for (int j = 0; j < d_embd; j++) {
            /// TODO: Use predefined pow fn.
            sum_squares += fp16_to_fp32(inp[i * d_embd + j]) * fp16_to_fp32(inp[i * d_embd + j]);
        }
        const float rms = sqrtf(sum_squares / (float)d_embd);
        const float rsqrt = 1.0f / (rms + eps);
        
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = fp32_to_fp16(fp16_to_fp32(inp[i * d_embd + j]) * rsqrt * fp16_to_fp32(weight[j]));
        }
        // x = xi / (root_mean_sq + 1e-6f) * wi
        // x = x / (rms+eps) * weight
    }
}


void rms_norm(const char* inp, const char* weight, char* out, int n_ctx, int d_embd, float eps, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            rms_norm_f16((Float16*)inp, (Float16*)weight, (Float16*)out, n_ctx, d_embd, eps);
            break;
        }
        case Dtype::Float32: {
            rms_norm_f32((float*)inp, (float*)weight, (float*)out, n_ctx, d_embd, eps);
            break;
        }
    }
}


// inp0: (n_ctx, d_embd)
// inp1: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void residual_f32(const float* inp0, const float* inp1, float* out, int n_ctx, int d_embd)
{
    for (int i = 0; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = inp0[i * d_embd + j] + inp1[i * d_embd + j];
        }
    }
}

void residual_f16(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_embd)
{
    for (int i = 0; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            out[i * d_embd + j] = fp32_to_fp16(fp16_to_fp32(inp0[i * d_embd + j]) + fp16_to_fp32(inp1[i * d_embd + j]));
        }
    }
}

void residual(const char* inp0, const char* inp1, char* out, int n_ctx, int d_embd, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            residual_f16((Float16*)inp0, (Float16*)inp1, (Float16*)out, n_ctx, d_embd);
            break;
        }
        case Dtype::Float32: {
            residual_f32((float*)inp0, (float*)inp1, (float*)out, n_ctx, d_embd);
            break;
        }
    }
}

// inp0: (n_ctx, d_embd)
// inp1: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void mul_inplace_f32(float* inp0, const float* inp1, int n_ctx, int d_embd, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            inp0[i * d_embd + j] = inp0[i * d_embd + j] * inp1[i * d_embd + j];
        }
    }
}

void mul_inplace_f16(Float16* inp0, const Float16* inp1, int n_ctx, int d_embd, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            inp0[i * d_embd + j] = fp32_to_fp16(fp16_to_fp32(inp0[i * d_embd + j]) * fp16_to_fp32(inp1[i * d_embd + j]));
        }
    }
}

void mul_inplace(char* inp0, const char* inp1, int n_ctx, int d_embd, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            mul_inplace_f16((Float16*)inp0, (Float16*)inp1, n_ctx, d_embd, start_pos);
            break;
        }
        case Dtype::Float32: {
            mul_inplace_f32((float*)inp0, (float*)inp1, n_ctx, d_embd, start_pos);
            break;
        }
    }
}

// Computes logits for next-token pred only.
// inp   : n_ctx, d_embd
// weight: n_vocab, d_embd
// out   : n_vocab 
void lm_head_proj_f32(const float* inp, const float* weight, float* out, int n_vocab, int n_ctx, int d_embd)
{
    for (int i = n_ctx - 1; i < n_ctx; i++) {
        for (int j = 0; j < n_vocab; j++) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_embd; k++) {
                dot_prod += inp[i * d_embd + k] * weight[j * d_embd + k];
            }
            out[j] = dot_prod;
        }
    }
}

void lm_head_proj_f16(const Float16* inp, const Float16* weight, float* out, int n_vocab, int n_ctx, int d_embd)
{
    for (int i = n_ctx - 1; i < n_ctx; i++) {
        for (int j = 0; j < n_vocab; j++) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_embd; k++) {
                dot_prod += fp16_to_fp32(inp[i * d_embd + k]) * fp16_to_fp32(weight[j * d_embd + k]);
            }
            out[j] = dot_prod;
        }
    }
}


void lm_head_proj(const char* inp, const char* weight, float* out, int n_vocab, int n_ctx, int d_embd, Dtype dtype)
{
    Timer timer{&metrics.matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            lm_head_proj_f16((Float16*)inp, (Float16*)weight, out, n_vocab, n_ctx, d_embd);
            break;
        }
        case Dtype::Float32: {
            lm_head_proj_f32((float*)inp, (float*)weight, out, n_vocab, n_ctx, d_embd);
            break;
        }
    }
}

// inp: (n_ctx, d_embd)
// out: (n_ctx, d_embd)
void silu_inplace_f32(float* inp, int n_ctx, int d_embd, int start_pos)
{
     for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            const float x = inp[i * d_embd + j];
            inp[i * d_embd + j] = x / (1.0f + expf(-x));
        }
    }
}

void silu_inplace_f16(Float16* inp, int n_ctx, int d_embd, int start_pos)
{
     for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < d_embd; j++) {
            const float x = fp16_to_fp32(inp[i * d_embd + j]);
            inp[i * d_embd + j] = fp32_to_fp16(x / (1.0f + expf(-x)));
        }
    }
}

void silu_inplace(char* inp, int n_ctx, int d_embd, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            silu_inplace_f16((Float16*)inp, n_ctx, d_embd, start_pos);
            break;
        }
        case Dtype::Float32: {
            silu_inplace_f32((float*)inp, n_ctx, d_embd, start_pos);
            break;
        }
    }
}

// inp0: (n_ctx, d_embd)
// inp1: (n_out, d_embd)
// out : (n_ctx, n_out)
void matmul_2d_f32(const float* inp0, const float* inp1, float* out, int n_ctx, int d_embd, int n_out, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < n_out; j++) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_embd; k++) {
                dot_prod += inp0[i * d_embd + k] * inp1[j * d_embd + k];
            }
            out[i * n_out + j] = dot_prod;
        }
    }   
}

void matmul_2d_f16(const Float16* inp0, const Float16* inp1, Float16* out, int n_ctx, int d_embd, int n_out, int start_pos)
{
    for (int i = start_pos; i < n_ctx; i++) {
        for (int j = 0; j < n_out; j++) {
            float dot_prod = 0.0f;
            for (int k = 0; k < d_embd; k++) {
                dot_prod += fp16_to_fp32(inp0[i * d_embd + k]) * fp16_to_fp32(inp1[j * d_embd + k]);
            }
            out[i * n_out + j] = fp32_to_fp16(dot_prod);
        }
    }   
}

void matmul_2d(const char* inp0, const char* inp1, char* out, int n_ctx, int d_embd, int n_out, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            matmul_2d_f16((Float16*)inp0, (Float16*)inp1, (Float16*)out, n_ctx, d_embd, n_out, start_pos);
            break;
        }
        case Dtype::Float32: {
            matmul_2d_f32((float*)inp0, (float*)inp1, (float*)out, n_ctx, d_embd, n_out, start_pos);
            break;
        }
    }
}

// q: (n_ctx, qn_embd) - (n_ctx, q_heads, d_head)[phy] -> (q_heads, n_ctx, d_head)[virt]
// k: (n_ctx, kn_embd) - (n_ctx, k_heads, d_head)[phy] -> (k_heads, n_ctx, d_head)[virt]
// out: (q_heads, n_ctx, n_ctx)
void qk_f32(const float* q, const float* k, float* out, int n_ctx, int q_heads, int kv_heads, int d_head, float scaler, int start_pos)
{
    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            for (int j = 0; j < n_ctx; j++) {
                float dot_prod = 0.0f;
                for (int kk = 0; kk < d_head; kk++) {
                    // index of the current head in k.
                    const int hk = h / q_group_size;
                    dot_prod += q[h * d_head + i * q_heads*d_head + kk] * k[hk * d_head + j * k_heads*d_head + kk];
                }
                out[h * n_ctx * n_ctx + i * n_ctx + j] = dot_prod * scaler;
            }
        }
    }
}


void qk_f16(const Float16* q, const Float16* k, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, float scaler, int start_pos)
{
    const int k_heads = kv_heads;
    // Note: In qroup query attn, we divide queries together into groups,
    // each of which share a single key and value.
    const int q_group_size = (int)(q_heads / k_heads);

    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            for (int j = 0; j < n_ctx; j++) {
                float dot_prod = 0.0f;
                for (int kk = 0; kk < d_head; kk++) {
                    // index of the current head in k.
                    const int hk = h / q_group_size;
                    dot_prod += fp16_to_fp32(q[h * d_head + i * q_heads*d_head + kk]) * fp16_to_fp32(k[hk * d_head + j * k_heads*d_head + kk]);
                }
                out[h * n_ctx * n_ctx + i * n_ctx + j] = fp32_to_fp16(dot_prod * scaler);
            }
        }
    }
}

void qk(const char* q, const char* k, char* out, int n_ctx, int q_heads, int kv_heads, int d_head, float scaler, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            qk_f16((Float16*)q, (Float16*)k, (Float16*)out, n_ctx, q_heads, kv_heads, d_head, scaler, start_pos);
            break;
        }
        case Dtype::Float32: {
             qk_f32((float*)q, (float*)k, (float*)out, n_ctx, q_heads, kv_heads, d_head, scaler, start_pos);
            break;
        }
    }
}

// inp: (n_heads, n_ctx, n_ctx)
void attn_mask_inplace_f32(float* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int i = start_pos; i < n_heads; i++) {
        for (int j = 0; j < n_ctx; j++) {
            const int start_ix = j + 1;
            for (int k = start_ix; k < n_ctx; k++) {
                inp[i * n_ctx * n_ctx + j * n_ctx + k] = -INFINITY;
            }
        }
    }
}

void attn_mask_inplace_f16(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int i = start_pos; i < n_heads; i++) {
        for (int j = 0; j < n_ctx; j++) {
            const int start_ix = j + 1;
            for (int k = start_ix; k < n_ctx; k++) {
                inp[i * n_ctx * n_ctx + j * n_ctx + k] = fp32_to_fp16(-INFINITY);
            }
        }
    }
}

void attn_mask_inplace(char* inp, int n_heads, int n_ctx, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            attn_mask_inplace_f16((Float16*)inp, n_heads, n_ctx, start_pos);
            break;
        }
        case Dtype::Float32: {
            attn_mask_inplace_f32((float*)inp, n_heads, n_ctx, start_pos);
            break;
        }
    }
}

// inp: [n_heads, n_ctx, n_ctz]
void softmax_inplace_f32(float* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int h = 0; h < n_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            float max = -INFINITY;
            for (int j = 0; j < n_ctx; j++) {
                const float val = inp[h * n_ctx * n_ctx + i * n_ctx + j];
                if (val > max) {
                    max = val;
                }
            }

            float sum_exp = 0;
            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                inp[idx] = expf(inp[idx] - max);
                sum_exp += inp[idx];
            }

            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                inp[idx] = inp[idx] / sum_exp;
            }
        }
    }
}

void softmax_inplace_f16(Float16* inp, int n_heads, int n_ctx, int start_pos)
{
    for (int h = 0; h < n_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            float max = -INFINITY;
            for (int j = 0; j < n_ctx; j++) {
                const float val = fp16_to_fp32(inp[h * n_ctx * n_ctx + i * n_ctx + j]);
                if (val > max) {
                    max = val;
                }
            }

            float sum_exp = 0;
            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                const float res = expf(fp16_to_fp32(inp[idx]) - max);
                sum_exp += res;
                inp[idx] = fp32_to_fp16(res);
            }

            for (int j = 0; j < n_ctx; j++) {
                const int idx = h * n_ctx * n_ctx + i * n_ctx + j;
                inp[idx] = fp32_to_fp16(fp16_to_fp32(inp[idx]) / sum_exp);
            }
        }
    }
}

void softmax_inplace(char* inp, int n_heads, int n_ctx, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            softmax_inplace_f16((Float16*)inp, n_heads, n_ctx, start_pos);
            break;
        }
        case Dtype::Float32: {
            softmax_inplace_f32((float*)inp, n_heads, n_ctx, start_pos);
            break;
        }
    }
}

// qk: (n_heads, n_ctx, n_ctx)
//  v: (n_ctx, vn_embd) - (n_ctx, v_heads, d_heads)[phy] - (v_heads, d_heads, n_ctx)[virt]
// out: (n_ctx, q_heads, d_head)
void qkv_f32(const float* qk, const float* v, float* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos)
{
    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);

    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            for (int j = 0; j < d_head; j++) {
                float dot_prod = 0.0f;
                for (int k = 0; k < n_ctx; k++) {
                    // index of the current head in v.
                    const int hv = h / qk_group_size;
                    dot_prod += qk[h * n_ctx*n_ctx + i * n_ctx + k] * v[hv * d_head + j + k * v_heads*d_head];
                }
                out[i * q_heads*d_head + h*d_head + j] = dot_prod;
            } 
        }
    }
}


void qkv_f16(const Float16* qk, const Float16* v, Float16* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos)
{
    const int v_heads = kv_heads;
    const int qk_group_size = (int)(q_heads / v_heads);

    for (int h = 0; h < q_heads; h++) {
        for (int i = start_pos; i < n_ctx; i++) {
            for (int j = 0; j < d_head; j++) {
                float dot_prod = 0.0f;
                for (int k = 0; k < n_ctx; k++) {
                    // index of the current head in v.
                    const int hv = h / qk_group_size;
                    dot_prod += fp16_to_fp32(qk[h * n_ctx*n_ctx + i * n_ctx + k]) * fp16_to_fp32(v[hv * d_head + j + k * v_heads*d_head]);
                }
                out[i * q_heads*d_head + h*d_head + j] = fp32_to_fp16(dot_prod);
            } 
        }
    }
}


void qkv(const char* qk, const char* v, char* out, int n_ctx, int q_heads, int kv_heads, int d_head, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            qkv_f16((Float16*)qk, (Float16*)v, (Float16*)out, n_ctx, q_heads, kv_heads, d_head, start_pos);
            break;
        }
        case Dtype::Float32: {
            qkv_f32((float*)qk, (float*)v, (float*)out, n_ctx, q_heads, kv_heads, d_head, start_pos);
            break;
        }
    }
}

// inp: [n_ctx, n_head, d_head]
void rotary_emb_f32(float* inp, int n_ctx, int n_heads, int d_head, int start_pos)
{
    for (int i = start_pos; i < n_ctx; ++i) {
       for (int h = 0; h < n_heads; ++h) {
            float* inp_vec = inp + i*n_heads*d_head + h*d_head;

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j) {
                const float x0 = inp_vec[j];
                const float x1 = inp_vec[j + d_half];
                
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

                inp_vec[j] = o0;
                inp_vec[j + d_half] = o1;
            }
        }
    }
}


void rotary_emb_f16(Float16* inp, int n_ctx, int n_heads, int d_head, int start_pos)
{
    for (int i = start_pos; i < n_ctx; ++i) {
       for (int h = 0; h < n_heads; ++h) {
            Float16* inp_vec = inp + i*n_heads*d_head + h*d_head;

            const int d_half = d_head / 2;
            for (int j = 0; j < d_half; ++j) {
                const float x0 = fp16_to_fp32(inp_vec[j]);
                const float x1 = fp16_to_fp32(inp_vec[j + d_half]);
                
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

                inp_vec[j] = fp32_to_fp16(o0);
                inp_vec[j + d_half] = fp32_to_fp16(o1);
            }
        }
    }
}


void rotary_emb(char* inp, int n_ctx, int n_heads, int d_head, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            rotary_emb_f16((Float16*)inp, n_ctx, n_heads, d_head, start_pos);
            break;
        }
        case Dtype::Float32: {
            rotary_emb_f32((float*)inp, n_ctx, n_heads, d_head, start_pos);
            break;
        }
    }
}
 

void copy_tensors(const char* src, char* dest, size_t size, Dtype dtype)
{
    Timer timer{&metrics.non_matmul_ms};

    switch (dtype) {
        case Dtype::Float16: {
            memcpy(dest, src, size*sizeof(Float16));
            break;
        }
        case Dtype::Float32: {
            memcpy(dest, src, size*sizeof(float));
            break;
        }
    }
}

} // namespace ops.