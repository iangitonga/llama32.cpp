#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <random>

#include "ops.h"
#include "tokenizer.h"
#include "metrics.h"


#define LLAMA32_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nLLAMA32_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }


struct Llama32Config
{
    uint32_t n_vocab;
    uint32_t n_layers;
    uint32_t d_embd;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t d_head;
    uint32_t d_mlp;
    float rms_norm_eps;
};


#define NUM_LAYERS 16
struct Llama32Weights
{
    char* emb_table;
    // blocks
    char* attn_norm[NUM_LAYERS];
    char* q_proj[NUM_LAYERS];
    char* k_proj[NUM_LAYERS];
    char* v_proj[NUM_LAYERS];
    char* o_proj[NUM_LAYERS];
    char* mlp_norm[NUM_LAYERS];
    char* gate_proj[NUM_LAYERS];
    char* up_proj[NUM_LAYERS];
    char* down_proj[NUM_LAYERS];

    char* out_norm;
};

struct Llama32Acvs
{
    char* emb_acv;
    char* attn_norm_acv[NUM_LAYERS];
    char* res_0_acv[NUM_LAYERS];
    char* res_1_acv[NUM_LAYERS];
    char* q_proj_acv[NUM_LAYERS];
    char* k_proj_acv[NUM_LAYERS];
    char* v_proj_acv[NUM_LAYERS];
    char* o_proj_acv[NUM_LAYERS];
    char* qk_acv[NUM_LAYERS];
    char* qkv_acv[NUM_LAYERS];
    char* mlp_norm_acv[NUM_LAYERS];
    char* mlp_gate_acv[NUM_LAYERS];
    char* mlp_up_acv[NUM_LAYERS];
    char* mlp_down_acv[NUM_LAYERS];
    char* out_norm_acv;
    float* logits_acv;
};


class Transformer
{
public:
    Dtype dtype;
    int max_ctx;
    Llama32Weights w;
    Llama32Acvs a;

    const Llama32Config config = {
        .n_vocab = 128256,
        .n_layers = 16,
        .d_embd = 2048,
        .n_heads = 32,
        .n_kv_heads = 8,
        .d_head = 64,
        .d_mlp = 8192,
        .rms_norm_eps = 1e-05f
    };

public:
    Transformer(Dtype dtype_, int max_ctx_)
        : dtype{dtype_}, max_ctx{max_ctx_}
    {}
};


size_t get_weights_nbytes(Transformer& t)
{
    size_t nbytes = 0;

    nbytes += t.config.n_vocab * t.config.d_embd;
    for (int i = 0; i < (int)t.config.n_layers; i++) {
        nbytes += t.config.d_embd;
        nbytes += t.config.n_heads    * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_kv_heads * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_kv_heads * t.config.d_head * t.config.d_embd;
        nbytes += t.config.n_heads    * t.config.d_head * t.config.d_embd;
        nbytes += t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
        nbytes += t.config.d_mlp * t.config.d_embd;
    }
    
    nbytes += t.config.d_embd;

    const int itemsize = t.dtype == Dtype::Float16 ? sizeof(Float16) : sizeof(float);
    nbytes = nbytes * itemsize;

    return nbytes;
}


size_t get_acvs_nbytes(Transformer& t)
{
    size_t nbytes = 0;

    nbytes += t.max_ctx * t.config.d_embd;
    for (int i = 0; i < (int)t.config.n_layers; i++) {
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.config.n_heads * t.max_ctx * t.max_ctx;
        nbytes += t.max_ctx * t.config.n_heads * t.config.d_head;
        nbytes += t.max_ctx * t.config.d_mlp;
        nbytes += t.max_ctx * t.config.d_mlp;
        nbytes += t.max_ctx * t.config.d_embd;
        nbytes += t.max_ctx * t.config.d_embd;
    }

    nbytes += t.max_ctx * t.config.d_embd;

    const int itemsize = t.dtype == Dtype::Float16 ? sizeof(Float16) : sizeof(float);
    nbytes = nbytes * itemsize;

    nbytes += t.config.n_vocab * sizeof(float); // Always float

    return nbytes;
}


void alloc_llama32_weights(char* ptr, Transformer& t)
{
    const int itemsize = t.dtype == Dtype::Float16 ? sizeof(Float16) : sizeof(float);

    const Llama32Config& c = t.config;

    t.w.emb_table = ptr;

    char* prev_layer_ptr = ptr + c.n_vocab * c.d_embd * itemsize;
    for (int i = 0; i < (int)c.n_layers; i++) {
        t.w.attn_norm[i] = prev_layer_ptr;
        t.w.q_proj[i]    = t.w.attn_norm[i] + c.d_embd * itemsize;
        t.w.k_proj[i]    = t.w.q_proj[i]    + c.n_heads * c.d_head * c.d_embd * itemsize;
        t.w.v_proj[i]    = t.w.k_proj[i]    + c.n_kv_heads * c.d_head * c.d_embd * itemsize;
        t.w.o_proj[i]    = t.w.v_proj[i]    + c.n_kv_heads * c.d_head * c.d_embd * itemsize;
        t.w.mlp_norm[i]  = t.w.o_proj[i]    + c.n_heads * c.d_head * c.d_embd * itemsize;
        t.w.gate_proj[i] = t.w.mlp_norm[i]  + c.d_embd * itemsize;
        t.w.up_proj[i]   = t.w.gate_proj[i] + c.d_mlp * c.d_embd * itemsize;
        t.w.down_proj[i] = t.w.up_proj[i]   + c.d_mlp * c.d_embd * itemsize;

        prev_layer_ptr = t.w.down_proj[i] + c.d_mlp * c.d_embd * itemsize;
    }
    
    t.w.out_norm = prev_layer_ptr;
}


void alloc_llama32_acvs(char* ptr, Transformer& t)
{
    const size_t itemsize = t.dtype == Dtype::Float16 ? sizeof(Float16) : sizeof(float);
    const Llama32Config& c = t.config;

    t.a.emb_acv = ptr;

    char* prev_layer_ptr = ptr + t.max_ctx * c.d_embd * itemsize;

    for (int i = 0; i < (int)c.n_layers; i++) {
        t.a.attn_norm_acv[i] = prev_layer_ptr;
        t.a.res_0_acv[i]     = t.a.attn_norm_acv[i] + t.max_ctx * c.d_embd * itemsize;
        t.a.res_1_acv[i]     = t.a.res_0_acv[i]     + t.max_ctx * c.d_embd * itemsize;
        t.a.q_proj_acv[i]    = t.a.res_1_acv[i]     + t.max_ctx * c.d_embd * itemsize;
        t.a.k_proj_acv[i]    = t.a.q_proj_acv[i]    + t.max_ctx * c.d_embd * itemsize;
        t.a.v_proj_acv[i]    = t.a.k_proj_acv[i]    + t.max_ctx * c.d_embd * itemsize;
        t.a.o_proj_acv[i]    = t.a.v_proj_acv[i]    + t.max_ctx * c.d_embd * itemsize;
        t.a.qk_acv[i]        = t.a.o_proj_acv[i]    + t.max_ctx * c.d_embd * itemsize;
        t.a.qkv_acv[i]       = t.a.qk_acv[i]        + c.n_heads * t.max_ctx * t.max_ctx * itemsize;
        t.a.mlp_gate_acv[i]  = t.a.qkv_acv[i]       + t.max_ctx * c.n_heads * c.d_head * itemsize;
        t.a.mlp_up_acv[i]    = t.a.mlp_gate_acv[i]  + t.max_ctx * c.d_mlp * itemsize;
        t.a.mlp_down_acv[i]  = t.a.mlp_up_acv[i]    + t.max_ctx * c.d_mlp * itemsize;
        t.a.mlp_norm_acv[i]  = t.a.mlp_down_acv[i]  + t.max_ctx * c.d_embd * itemsize;

        prev_layer_ptr = t.a.mlp_norm_acv[i] + t.max_ctx * c.d_embd * itemsize;
    }

    t.a.out_norm_acv  = prev_layer_ptr;
    t.a.logits_acv    = (float*)(t.a.out_norm_acv + t.max_ctx * c.d_embd * itemsize); // Always float
}


void alloc_llama32(Transformer& t)
{
    const size_t weights_nbytes = get_weights_nbytes(t);
    const size_t acvs_nbytes = get_acvs_nbytes(t);
    const size_t total_nbytes = weights_nbytes + acvs_nbytes;

    metrics.total_mem_bytes += total_nbytes;
    char* memptr = reinterpret_cast<char*>(std::malloc(total_nbytes));
    if (!memptr) {
        std::fprintf(stderr, "%s: Failed to allocate %ld bytes.\n", __func__, total_nbytes);
        std::exit(-1);
    }

    alloc_llama32_weights(memptr, t);
    alloc_llama32_acvs(memptr + weights_nbytes, t);
}

void free_llama32(Transformer& t)
{
    std::free(t.w.emb_table);
}


void load_llama32_weights(const char* fpath, Transformer& t)
{
    Timer timer{&metrics.load_time_ms};

    std::FILE* fin = std::fopen(fpath, "rb");

    if (!fin) {
        std::fprintf(stderr, "%s: failed to open %s.\n", __func__, fpath);
        std::exit(-1);
    }

    const int64_t true_magic_no = 0x663233616d616c6c; // Hex for ASCII string: llama32f
    int64_t magic_no;
    LLAMA32_ASSERT(fread(&magic_no, sizeof(int64_t), 1, fin) == 1);

    if (magic_no != true_magic_no) {
        fprintf(stderr, "Magic number: %ld failed to match the expected one: %ld.\n", magic_no, true_magic_no);
        fclose(fin);
        exit(-1);
    }

    const size_t weights_nbytes = get_weights_nbytes(t);

    LLAMA32_ASSERT(fread(t.w.emb_table, weights_nbytes, 1, fin) == 1);

    fclose(fin);
}


float* forward(Transformer& t, const int* tokens, int n_ctx, int start_pos)
{
    Timer timer{&metrics.inference_time_ms};

    const Llama32Config& c = t.config;

    ops::embed(tokens, t.w.emb_table, t.a.emb_acv, c.n_vocab, n_ctx, c.d_embd, start_pos, t.dtype);

    char* next_layer_inp = t.a.emb_acv;

    for (int i = 0; i < (int)c.n_layers; i++) {
        ops::copy_tensors(next_layer_inp, t.a.res_0_acv[i], n_ctx, c.d_embd, start_pos, t.dtype);

        ops::rms_norm(next_layer_inp, t.w.attn_norm[i], t.a.attn_norm_acv[i], n_ctx, c.d_embd, start_pos, c.rms_norm_eps, t.dtype);

        // ATTN
        // [n_ctx, n_emb], [d_out, d_embd]
        const int q_dim = c.n_heads * c.d_head;
        const int kv_dim = c.n_kv_heads * c.d_head;
        ops::matmul_2d(t.a.attn_norm_acv[i], t.w.q_proj[i], t.a.q_proj_acv[i], n_ctx, c.d_embd, q_dim, start_pos, t.dtype);
        ops::matmul_2d(t.a.attn_norm_acv[i], t.w.k_proj[i], t.a.k_proj_acv[i], n_ctx, c.d_embd, kv_dim, start_pos, t.dtype);
        ops::matmul_2d(t.a.attn_norm_acv[i], t.w.v_proj[i], t.a.v_proj_acv[i], n_ctx, c.d_embd, kv_dim, start_pos, t.dtype);
        ops::rotary_emb(t.a.q_proj_acv[i], n_ctx, c.n_heads, c.d_head, start_pos, t.dtype);
        ops::rotary_emb(t.a.k_proj_acv[i], n_ctx, c.n_kv_heads, c.d_head, start_pos, t.dtype);

        const float qk_scaler = 1.0f / sqrtf(c.d_head);
        ops::qk(t.a.q_proj_acv[i], t.a.k_proj_acv[i], t.a.qk_acv[i], n_ctx, c.n_heads, c.n_kv_heads, c.d_head, qk_scaler, start_pos, t.dtype);
        ops::attn_mask_inplace(t.a.qk_acv[i], c.n_heads, n_ctx, start_pos, t.dtype);
        ops::softmax_inplace(t.a.qk_acv[i], c.n_heads, n_ctx, start_pos, t.dtype);
        ops::qkv(t.a.qk_acv[i], t.a.v_proj_acv[i], t.a.qkv_acv[i], n_ctx, c.n_heads, c.n_kv_heads, c.d_head, start_pos, t.dtype);
        ops::matmul_2d(t.a.qkv_acv[i], t.w.o_proj[i], t.a.o_proj_acv[i], n_ctx, c.d_embd, c.d_embd, start_pos, t.dtype);

        ops::residual(t.a.o_proj_acv[i], t.a.res_0_acv[i], t.a.res_1_acv[i], n_ctx, c.d_embd, start_pos, t.dtype);

        // MLP
        // self.w2(F.silu(self.w1(x)) * self.w3(x))
        // down(silu(gate(x)) * up(x))
        ops::rms_norm(t.a.res_1_acv[i], t.w.mlp_norm[i], t.a.mlp_norm_acv[i], n_ctx, c.d_embd, start_pos, c.rms_norm_eps, t.dtype);
        ops::matmul_2d(t.a.mlp_norm_acv[i], t.w.gate_proj[i], t.a.mlp_gate_acv[i], n_ctx, c.d_embd, c.d_mlp, start_pos, t.dtype);
        ops::matmul_2d(t.a.mlp_norm_acv[i], t.w.up_proj[i], t.a.mlp_up_acv[i], n_ctx, c.d_embd, c.d_mlp, start_pos, t.dtype);

        ops::silu_inplace(t.a.mlp_gate_acv[i], n_ctx, c.d_mlp, start_pos, t.dtype);
        ops::mul_inplace(t.a.mlp_gate_acv[i], t.a.mlp_up_acv[i], n_ctx, c.d_mlp, start_pos, t.dtype);
        ops::matmul_2d(t.a.mlp_gate_acv[i], t.w.down_proj[i], t.a.mlp_down_acv[i], n_ctx, c.d_mlp, c.d_embd, start_pos, t.dtype);

        ops::residual(t.a.res_1_acv[i], t.a.mlp_down_acv[i], t.a.res_1_acv[i], n_ctx, c.d_embd, start_pos, t.dtype);

        next_layer_inp = t.a.res_1_acv[i];
    }

    ops::rms_norm(next_layer_inp, t.w.out_norm, t.a.out_norm_acv, n_ctx, c.d_embd, start_pos, c.rms_norm_eps, t.dtype);
    
    ops::lm_head_proj(t.a.out_norm_acv, t.w.emb_table, t.a.logits_acv, c.n_vocab, n_ctx, c.d_embd, t.dtype);

    return t.a.logits_acv;
}


int topk_sample(Transformer& t, Llama32Tokenizer& tokenizer, const std::string& prompt, int top_k, float top_p, float temp)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    if ((int)tokens.size() >= t.max_ctx) {
        std::fprintf(stderr, "Prompt is too large: %d for max context size: %d\n", (int)tokens.size(), t.max_ctx);
        return 0;
    }

    const int logits_size = t.config.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = tokenizer.eot_id;

    const int n_pred_tokens = t.max_ctx - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++) {
        const int start_pos = i == 0 ? 0 : tokens.size()-1;
        const float* logits = forward(t, tokens.data(), tokens.size(), start_pos);

        Timer sample_timer{&metrics.sample_time_ms};

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j) {
            logits_probs.push_back(std::make_pair((double)logits[j] / temp, j));
        }
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j) {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j) {
            logits_probs[j].first = logits_probs[j].first / sum_exp;
        }

        // top_p selection
        int top_p_count = top_k;
        double cumulative_prob = 0.0f;
        for (int j = 0; j < top_k; j++) {
            cumulative_prob += logits_probs[j].first;
            if (cumulative_prob >= top_p) {
                top_p_count = j + 1;
                break;
            }
        }

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_p_count; j++) {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        const int pred_token = dist(gen);
        if (pred_token == eot_token) {
            // printf("<EOT: %d>\n", eot_token);
            break;
        }

        std::printf("%s", tokenizer.decode(pred_token));
        std::fflush(stdout);

        tokens.push_back(pred_token);
    }
    printf("\n");

    return tokens.size();
}


static const char *usage_message = R"(
USAGE:
./llama32 [options] -p PROMPT  for a single prompt or
./llama32 [options] for a chat interface. 

Optional args. 
-f16 :     Use float-16 model (2.3GB). [default]
--npred  N : Max context size. Minimum is 128 and max is 8192 [default=512]. Higher values consume more memory.
)";


int main(int argc, char const *argv[])
{
    Timer timer{&metrics.total_runtime_ms};

    const char* model_path = "models/llama32-1B.fp16.bin";
    const Dtype model_dtype = Dtype::Float16;
    int max_ctx = 512;
    std::string prompt = "";

    for (int i = 1; i < argc; i++) {
        std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            fprintf(stderr, "%s\n.", usage_message);
            return 0;
        }
        else if (arg == "-f16") {
            continue;
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                fprintf(stderr, "error: Prompt not provided.\n");
                fprintf(stderr, "%s\n.", usage_message);
                return -1;
            }
        }
        else if (arg == "--npred") {
            if (argc <= i+1) {
                fprintf(stderr, "npred value is missing.\n");
                return -1;
            }
            int npred;
            try {
                npred = std::stoi(argv[i+1]);
            } catch (...) {
                fprintf(stderr, "Invalid npred value.\n");
                return -1;
            }
            if (npred < 128 || npred > 8192) {
                fprintf(stderr, "npred must be greater than 128 and less than 2048.\n");
                return -1;
            }
            max_ctx = npred;
            i += 1; // skip len param
        }
        else {
            fprintf(stderr, "error: Unknown argument: %s\n", arg.data());
            fprintf(stderr, "%s\n.", usage_message);
            return -1;
        }
    }

#ifdef _WIN32
    int res = std::system("python model_dl.py");
#else
    int res = std::system("python3 model_dl.py");
#endif
    if (res != 0) {
        fprintf(stderr, "Error: Failed to download the model. Check your network connectivity.\n");
        return -1;
    }

    Transformer model{model_dtype, max_ctx};

    alloc_llama32(model);
    load_llama32_weights(model_path, model);

    // size of the vocab without special tokens eg <|start_of_text|>.
    const int vocab_tok_size = 128000;
    Llama32Tokenizer tokenizer{"tokenizer.bin", vocab_tok_size};

    const int top_k = 40;
    const float top_p = 0.95;
    const float temp = 0.8;

    if (prompt == "") {
        printf("Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n");
        std::string prompt;
        while (true) {
            printf("\n\n[You]: "); fflush(stdout);

            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            printf("\n\n[LLAMA-1B]: \n"); fflush(stdout);
            
            topk_sample(model, tokenizer, prompt, top_k, top_p, temp);
        } 
    } else {
        printf("\n[PROMPT]:\n%s\n\n[LLAMA-1B]: ", prompt.c_str());
        std::fflush(stdout);

        const int processed_toks = topk_sample(model, tokenizer, prompt, top_k, top_p, temp);
        timer.stop();
        print_metrics(metrics, processed_toks);
    }

    free_llama32(model);

    return 0;
}

