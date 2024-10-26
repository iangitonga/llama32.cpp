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

static const struct Llama32Config cfg = {
    .n_vocab = 128256,
    .n_layers = 16,
    .d_embd = 2048,
    .n_heads = 32,
    .n_kv_heads = 8,
    .d_head = 64,
    .d_mlp = 8192,
    .rms_norm_eps = 1e-05f
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


char* llama_malloc(size_t nbytes, Dtype dtype)
{
    nbytes = nbytes;
    if (dtype == Dtype::Float16) {
        nbytes *= sizeof(Float16);
    } else {
        nbytes *= sizeof(float);
    }

    metrics.total_mem_bytes += nbytes;
    void* allocated = std::malloc(nbytes);
    if (!allocated) {
        std::fprintf(stderr, "Failed to allocate %ld bytes.\n", nbytes);
        std::exit(-1);
    }

    return reinterpret_cast<char*>(allocated);
}

void alloc_llama32_weights(struct Llama32Weights* w, Dtype dtype)
{
    w->emb_table = llama_malloc(cfg.n_vocab * cfg.d_embd, dtype);
    for (int i = 0; i < (int)cfg.n_layers; i++) {
        w->attn_norm[i] = llama_malloc(cfg.d_embd, dtype);
        w->q_proj[i]    = llama_malloc(cfg.n_heads * cfg.d_head * cfg.d_embd, dtype);
        w->k_proj[i]    = llama_malloc(cfg.n_kv_heads * cfg.d_head * cfg.d_embd, dtype);
        w->v_proj[i]    = llama_malloc(cfg.n_kv_heads * cfg.d_head * cfg.d_embd, dtype);
        w->o_proj[i]    = llama_malloc(cfg.n_heads * cfg.d_head * cfg.d_embd, dtype);
        w->mlp_norm[i]  = llama_malloc(cfg.d_embd, dtype);
        w->gate_proj[i] = llama_malloc(cfg.d_mlp * cfg.d_embd, dtype);
        w->up_proj[i]   = llama_malloc(cfg.d_mlp * cfg.d_embd, dtype);
        w->down_proj[i] = llama_malloc(cfg.d_mlp * cfg.d_embd, dtype);
    }
    
    w->out_norm = llama_malloc(cfg.d_embd, dtype);
}

void free_llama32_weights(struct Llama32Weights* w)
{
    std::free(w->emb_table);
    for (int i = 0; i < (int)cfg.n_layers; i++) {
        std::free(w->attn_norm[i]);
        std::free(w->q_proj[i]);   
        std::free(w->k_proj[i]); 
        std::free(w->v_proj[i]);
        std::free(w->o_proj[i]);   
        std::free(w->mlp_norm[i]);
        std::free(w->gate_proj[i]);
        std::free(w->up_proj[i]);
        std::free(w->down_proj[i]);
    }
    
    std::free(w->out_norm);
}


#define LLAMA32_ASSERT(condition)  \
    if (!(condition)) { \
        std::fprintf(stderr, "\nLLAMA32_ASSERT: %s:%d: %s.\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    }

void init_llama32_weights(const char* fpath, Llama32Weights* w, Dtype dtype)
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

    const struct Llama32Config& c = cfg;
    int itemsize;
    if (dtype == Dtype::Float16) { itemsize = sizeof(Float16); }
    else { itemsize = sizeof(float); }

    LLAMA32_ASSERT(fread(w->emb_table, itemsize, c.n_vocab*c.d_embd, fin) == c.n_vocab*c.d_embd);

    for (int i = 0; i < (int)cfg.n_layers; i++) {
        LLAMA32_ASSERT(fread(w->attn_norm[i], itemsize, c.d_embd, fin) == c.d_embd);
        LLAMA32_ASSERT(fread(w->q_proj[i], itemsize, c.n_heads * c.d_head * c.d_embd, fin) == c.n_heads * c.d_head * c.d_embd);
        LLAMA32_ASSERT(fread(w->k_proj[i], itemsize, c.n_kv_heads * c.d_head * c.d_embd, fin) == c.n_kv_heads * c.d_head * c.d_embd);
        LLAMA32_ASSERT(fread(w->v_proj[i], itemsize, c.n_kv_heads * c.d_head * c.d_embd, fin) == c.n_kv_heads * c.d_head * c.d_embd);
        LLAMA32_ASSERT(fread(w->o_proj[i], itemsize, c.n_heads * c.d_head * c.d_embd, fin) == c.n_heads * c.d_head * c.d_embd);
        LLAMA32_ASSERT(fread(w->mlp_norm[i], itemsize, c.d_embd, fin) == c.d_embd);
        LLAMA32_ASSERT(fread(w->gate_proj[i], itemsize, c.d_mlp * c.d_embd, fin) == c.d_mlp * c.d_embd);
        LLAMA32_ASSERT(fread(w->up_proj[i], itemsize, c.d_mlp * c.d_embd, fin) == c.d_mlp * c.d_embd);
        LLAMA32_ASSERT(fread(w->down_proj[i], itemsize, c.d_mlp * c.d_embd, fin) == c.d_mlp * c.d_embd);
    }
    
    LLAMA32_ASSERT(fread(w->out_norm, itemsize, c.d_embd, fin) == c.d_embd);

    fclose(fin);
}

struct Llama32Acvs
{
    Timer timer{&metrics.load_time_ms};

    char* emb_acv;
    char* attn_norm_acv;
    char* res_0_acv;
    char* res_1_acv;
    char* q_proj_acv[NUM_LAYERS];
    char* k_proj_acv[NUM_LAYERS];
    char* v_proj_acv[NUM_LAYERS];
    char* o_proj_acv[NUM_LAYERS];
    char* qk_acv[NUM_LAYERS];
    char* qkv_acv[NUM_LAYERS];
    char* mlp_norm_acv;
    char* mlp_gate_acv[NUM_LAYERS];
    char* mlp_up_acv[NUM_LAYERS];
    char* mlp_down_acv[NUM_LAYERS];
    char* out_norm_acv;
    float* logits_acv;
};


// f32[256]: 779MB  - 0.4GB
// f32[512]: 1.8GB  - 0.9GB
// f32[768]: 3.1GB  - 1.6GB
// f32[1024]: 4.7GB - 2.3GB

// f32[256]: 0.6GB  - 0.4GB
// f32[512]: 1.5GB  - 0.7GB
// f32[768]: 2.6GB  - 1.3GB
// f32[1024]:4.0GB  - 2.0GB


void alloc_llama32_acvs(struct Llama32Acvs* a, Dtype dtype, int max_ctx)
{
    Timer timer{&metrics.load_time_ms};
    // max_ctx*n_embd *  10 + max_ctx*max_ctx*n_heads + max_ctx*n_head*d_head + max_ctx*d_mlp * 2 + n_vocab

    a->emb_acv       = llama_malloc(max_ctx * cfg.d_embd, dtype);
    a->attn_norm_acv = llama_malloc(max_ctx * cfg.d_embd, dtype);
    a->res_0_acv     = llama_malloc(max_ctx * cfg.d_embd, dtype);
    a->res_1_acv     = llama_malloc(max_ctx * cfg.d_embd, dtype);
    for (int i = 0; i < (int)cfg.n_layers; i++) {
        a->q_proj_acv[i]    = llama_malloc(max_ctx * cfg.d_embd, dtype);
        a->k_proj_acv[i]    = llama_malloc(max_ctx * cfg.d_embd, dtype);
        a->v_proj_acv[i]    = llama_malloc(max_ctx * cfg.d_embd, dtype);
        a->o_proj_acv[i]    = llama_malloc(max_ctx * cfg.d_embd, dtype);
        a->qk_acv[i]        = llama_malloc(cfg.n_heads * max_ctx * max_ctx, dtype);
        a->qkv_acv[i]       = llama_malloc(max_ctx * cfg.n_heads * cfg.d_head, dtype);
        a->mlp_gate_acv[i]  = llama_malloc(max_ctx * cfg.d_mlp, dtype);
        a->mlp_up_acv[i]    = llama_malloc(max_ctx * cfg.d_mlp, dtype);
        a->mlp_down_acv[i]  = llama_malloc(max_ctx * cfg.d_embd, dtype);
    }
    a->mlp_norm_acv  = llama_malloc(max_ctx * cfg.d_embd, dtype);
    a->out_norm_acv  = llama_malloc(max_ctx * cfg.d_embd, dtype);
    a->logits_acv    = (float*)llama_malloc(cfg.n_vocab, Dtype::Float32); // Always float
}

void free_llama32_acvs(struct Llama32Acvs* a)
{
    std::free(a->emb_acv);
    std::free(a->attn_norm_acv);
    std::free(a->res_0_acv);
    std::free(a->res_1_acv);
    for (int i = 0; i < (int)cfg.n_layers; i++) {
        std::free(a->q_proj_acv[i]);
        std::free(a->k_proj_acv[i]);
        std::free(a->v_proj_acv[i]);
        std::free(a->o_proj_acv[i]);
        std::free(a->qk_acv[i]);
        std::free(a->qkv_acv[i]);
        std::free(a->mlp_gate_acv[i]);
        std::free(a->mlp_up_acv[i]);
        std::free(a->mlp_down_acv[i]);
    }
    
    std::free(a->mlp_norm_acv);
    std::free(a->out_norm_acv);
    std::free(a->logits_acv);
}

float* forward(const int* tokens, int n_ctx, const struct Llama32Weights* w, struct Llama32Acvs* a, int start_pos, Dtype dtype)
{
    Timer timer{&metrics.inference_time_ms};

    ops::embed(tokens, w->emb_table, a->emb_acv, cfg.n_vocab, n_ctx, cfg.d_embd, dtype);

    char* next_layer_inp = a->emb_acv;

    for (int i = 0; i < (int)cfg.n_layers; i++) {
        ops::copy_tensors(next_layer_inp, a->res_0_acv, n_ctx*cfg.d_embd, dtype);

        ops::rms_norm(next_layer_inp, w->attn_norm[i], a->attn_norm_acv, n_ctx, cfg.d_embd, cfg.rms_norm_eps, dtype);

        // ATTN
        // [n_ctx, n_emb], [d_out, d_embd]
        const int q_dim = cfg.n_heads * cfg.d_head;
        const int kv_dim = cfg.n_kv_heads * cfg.d_head;
        ops::matmul_2d(a->attn_norm_acv, w->q_proj[i], a->q_proj_acv[i], n_ctx, cfg.d_embd, q_dim, start_pos, dtype);
        ops::matmul_2d(a->attn_norm_acv, w->k_proj[i], a->k_proj_acv[i], n_ctx, cfg.d_embd, kv_dim, start_pos, dtype);
        ops::matmul_2d(a->attn_norm_acv, w->v_proj[i], a->v_proj_acv[i], n_ctx, cfg.d_embd, kv_dim, start_pos, dtype);
        ops::rotary_emb(a->q_proj_acv[i], n_ctx, cfg.n_heads, cfg.d_head, start_pos, dtype);
        ops::rotary_emb(a->k_proj_acv[i], n_ctx, cfg.n_kv_heads, cfg.d_head, start_pos, dtype);

        const float qk_scaler = 1.0f / sqrtf(cfg.d_head);
        ops::qk(a->q_proj_acv[i], a->k_proj_acv[i], a->qk_acv[i], n_ctx, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, qk_scaler, start_pos, dtype);
        ops::attn_mask_inplace(a->qk_acv[i], cfg.n_heads, n_ctx, start_pos, dtype);
        ops::softmax_inplace(a->qk_acv[i], cfg.n_heads, n_ctx, start_pos, dtype);
        ops::qkv(a->qk_acv[i], a->v_proj_acv[i], a->qkv_acv[i], n_ctx, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, start_pos, dtype);
        ops::matmul_2d(a->qkv_acv[i], w->o_proj[i], a->o_proj_acv[i], n_ctx, cfg.d_embd, cfg.d_embd, start_pos, dtype);

        ops::residual(a->o_proj_acv[i], a->res_0_acv, a->res_1_acv, n_ctx, cfg.d_embd, dtype);

        // MLP
        // self.w2(F.silu(self.w1(x)) * self.w3(x))
        // down(silu(gate(x)) * up(x))
        ops::rms_norm(a->res_1_acv, w->mlp_norm[i], a->mlp_norm_acv, n_ctx, cfg.d_embd, cfg.rms_norm_eps, dtype);
        ops::matmul_2d(a->mlp_norm_acv, w->gate_proj[i], a->mlp_gate_acv[i], n_ctx, cfg.d_embd, cfg.d_mlp, start_pos, dtype);
        ops::matmul_2d(a->mlp_norm_acv, w->up_proj[i], a->mlp_up_acv[i], n_ctx, cfg.d_embd, cfg.d_mlp, start_pos, dtype);

        ops::silu_inplace(a->mlp_gate_acv[i], n_ctx, cfg.d_mlp, start_pos, dtype);
        ops::mul_inplace(a->mlp_gate_acv[i], a->mlp_up_acv[i], n_ctx, cfg.d_mlp, start_pos, dtype);
        ops::matmul_2d(a->mlp_gate_acv[i], w->down_proj[i], a->mlp_down_acv[i], n_ctx, cfg.d_mlp, cfg.d_embd, start_pos, dtype);

        ops::residual(a->res_1_acv, a->mlp_down_acv[i], a->res_1_acv, n_ctx, cfg.d_embd, dtype);

        next_layer_inp =a->res_1_acv;
    }

    ops::rms_norm(next_layer_inp, w->out_norm, a->out_norm_acv, n_ctx, cfg.d_embd, cfg.rms_norm_eps, dtype);
    
    ops::lm_head_proj(a->out_norm_acv, w->emb_table, a->logits_acv, cfg.n_vocab, n_ctx, cfg.d_embd, dtype);

    return a->logits_acv;
}


int topk_sample(const std::string& prompt, Llama32Weights* w, Llama32Acvs* a, Llama32Tokenizer& tokenizer, Dtype dtype, int max_ctx, const float temp, const int top_k)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    if ((int)tokens.size() >= max_ctx) {
        std::fprintf(stderr, "Prompt is too large: %d for max context size: %d\n", (int)tokens.size(), max_ctx);
        return 0;
    }

    const int logits_size = cfg.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = tokenizer.eot_id;

    const int n_pred_tokens = max_ctx - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++) {
        const int start_pos = i == 0 ? 0 : tokens.size()-1;
        const float* logits = forward(tokens.data(), tokens.size(), w, a, start_pos, dtype);

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
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
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

    struct Llama32Weights w;
    struct Llama32Acvs a;

    alloc_llama32_weights(&w, model_dtype);
    alloc_llama32_acvs(&a, model_dtype, max_ctx);

    init_llama32_weights(model_path, &w, model_dtype);

    // size of the vocab without special tokens eg <|start_of_text|>.
    const int vocab_tok_size = 128000;
    Llama32Tokenizer tokenizer{"tokenizer.bin", vocab_tok_size};

    if (prompt == "") {
        printf("Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n");
        std::string prompt;
        while (true) {
            printf("\n\n[You]: "); fflush(stdout);

            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            printf("\n\n[LLAMA-1B]: \n"); fflush(stdout);
            
            topk_sample(prompt, &w, &a, tokenizer, model_dtype, max_ctx, 0.9f, 40);
        } 
    } else {
        printf("\n[PROMPT]:\n%s\n\n[LLAMA-1B]: ", prompt.c_str());
        std::fflush(stdout);

        const int processed_toks = topk_sample(prompt, &w, &a, tokenizer, model_dtype, max_ctx, 0.9f, 40);
        timer.stop();
        print_metrics(metrics, processed_toks);
    }

    free_llama32_weights(&w);
    free_llama32_acvs(&a);

    return 0;
}
