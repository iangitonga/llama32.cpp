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
    int n_vocab;
    int n_layers;
    int d_embd;
    int n_heads;
    int n_kv_heads;
    int d_head;
    int d_mlp;
};

struct LayerWeights {
    char* attn_norm;
    char* q_proj;
    char* k_proj;
    char* v_proj;
    char* o_proj;
    char* mlp_norm;
    char* gate_proj;
    char* up_proj;
    char* down_proj;
};

#define NUM_LAYERS 16
struct Llama32Weights
{
    char* emb_table;
    // blocks
    LayerWeights layers[NUM_LAYERS];
    char* out_norm;
};

struct LayerAcvs
{
    char* attn_norm_acv;
    char* res_0_acv;
    char* res_1_acv;
    char* q_proj_acv;
    char* k_proj_acv;
    char* v_proj_acv;
    char* o_proj_acv;
    char* qk_acv;
    char* qkv_acv;
    char* mlp_norm_acv;
    char* mlp_gate_acv;
    char* mlp_up_acv;
    char* mlp_down_acv;
};

struct Llama32Acvs
{
    char* emb_acv;
    LayerAcvs layers[NUM_LAYERS];
    char* out_norm_acv;
    float* logits_acv;
};


class Transformer
{
public:
    Device device;
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
        .d_mlp = 8192
    };

public:
    Transformer(Device device_, int max_ctx_)
        : device{device_}, max_ctx{max_ctx_}
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

    const int itemsize = sizeof(Float16);
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

    const int itemsize = sizeof(Float16);
    nbytes = nbytes * itemsize;

    nbytes += t.config.n_vocab * sizeof(float); // Always float

    return nbytes;
}


void alloc_llama32_weights(char* ptr, Transformer& t)
{
    const Llama32Config& c = t.config;
    const int itemsize = sizeof(Float16);

    t.w.emb_table = ptr;

    char* prev_layer_ptr = ptr + c.n_vocab * c.d_embd * itemsize;
    for (int i = 0; i < (int)c.n_layers; i++) {
        t.w.layers[i].attn_norm = prev_layer_ptr;
        t.w.layers[i].q_proj    = t.w.layers[i].attn_norm + c.d_embd * itemsize;
        t.w.layers[i].k_proj    = t.w.layers[i].q_proj    + c.n_heads * c.d_head * c.d_embd * itemsize;
        t.w.layers[i].v_proj    = t.w.layers[i].k_proj    + c.n_kv_heads * c.d_head * c.d_embd * itemsize;
        t.w.layers[i].o_proj    = t.w.layers[i].v_proj    + c.n_kv_heads * c.d_head * c.d_embd * itemsize;
        t.w.layers[i].mlp_norm  = t.w.layers[i].o_proj    + c.n_heads * c.d_head * c.d_embd * itemsize;
        t.w.layers[i].gate_proj = t.w.layers[i].mlp_norm  + c.d_embd * itemsize;
        t.w.layers[i].up_proj   = t.w.layers[i].gate_proj + c.d_mlp * c.d_embd * itemsize;
        t.w.layers[i].down_proj = t.w.layers[i].up_proj   + c.d_mlp * c.d_embd * itemsize;

        prev_layer_ptr = t.w.layers[i].down_proj + c.d_mlp * c.d_embd * itemsize;
    }
    
    t.w.out_norm = prev_layer_ptr;
}


void alloc_llama32_acvs(char* ptr, Transformer& t)
{
    const Llama32Config& c = t.config;
    const size_t itemsize = sizeof(Float16);

    t.a.emb_acv = ptr;

    char* prev_layer_ptr = ptr + t.max_ctx * c.d_embd * itemsize;

    for (int i = 0; i < (int)c.n_layers; i++) {
        t.a.layers[i].attn_norm_acv = prev_layer_ptr;
        t.a.layers[i].res_0_acv     = t.a.layers[i].attn_norm_acv + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].res_1_acv     = t.a.layers[i].res_0_acv     + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].q_proj_acv    = t.a.layers[i].res_1_acv     + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].k_proj_acv    = t.a.layers[i].q_proj_acv    + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].v_proj_acv    = t.a.layers[i].k_proj_acv    + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].o_proj_acv    = t.a.layers[i].v_proj_acv    + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].qk_acv        = t.a.layers[i].o_proj_acv    + t.max_ctx * c.d_embd * itemsize;
        t.a.layers[i].qkv_acv       = t.a.layers[i].qk_acv        + c.n_heads * t.max_ctx * t.max_ctx * itemsize;
        t.a.layers[i].mlp_gate_acv  = t.a.layers[i].qkv_acv       + t.max_ctx * c.n_heads * c.d_head * itemsize;
        t.a.layers[i].mlp_up_acv    = t.a.layers[i].mlp_gate_acv  + t.max_ctx * c.d_mlp * itemsize;
        t.a.layers[i].mlp_down_acv  = t.a.layers[i].mlp_up_acv    + t.max_ctx * c.d_mlp * itemsize;
        t.a.layers[i].mlp_norm_acv  = t.a.layers[i].mlp_down_acv  + t.max_ctx * c.d_embd * itemsize;

        prev_layer_ptr = t.a.layers[i].mlp_norm_acv + t.max_ctx * c.d_embd * itemsize;
    }

    t.a.out_norm_acv  = prev_layer_ptr;
    t.a.logits_acv    = (float*)(t.a.out_norm_acv + t.max_ctx * c.d_embd * itemsize); // Always float
}

char* llama32_malloc(size_t size, Device device) {
    if (device == Device::CPU) {
        char* memptr = reinterpret_cast<char*>(std::malloc(size));
        if (!memptr) {
            std::fprintf(stderr, "%s: Failed to allocate %ld bytes.\n", __func__, size);
            std::exit(-1);
        }
        return memptr;
    } else {
#if defined(__NVCC__)
        char* memptr;
        if (cudaMalloc(&memptr, size) != cudaSuccess) {
            std::fprintf(stderr, "%s: Failed to allocate %ld bytes on the GPU.\n", __func__, size);
            std::exit(-1);
        }    
        return memptr;
#else
        return nullptr;
#endif
    }
}

void llama32_free(void* ptr, Device device) {
    if (device == Device::CPU) {
        free(ptr);
    } else {
#if defined(__NVCC__)
        cudaFree(ptr);
#endif
    }
}

void alloc_llama32(Transformer& t)
{
    const size_t weights_nbytes = get_weights_nbytes(t);
    const size_t acvs_nbytes = get_acvs_nbytes(t);
    const size_t total_nbytes = weights_nbytes + acvs_nbytes;

    metrics.total_mem_bytes += total_nbytes;
    char* memptr = llama32_malloc(total_nbytes, t.device);

    alloc_llama32_weights(memptr, t);
    alloc_llama32_acvs(memptr + weights_nbytes, t);
}

void free_llama32(Transformer& t)
{
    llama32_free(t.w.emb_table, t.device);
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

    if (t.device == Device::CPU) {
        LLAMA32_ASSERT(fread(t.w.emb_table, weights_nbytes, 1, fin) == 1);
    } else {
        char* host = llama32_malloc(weights_nbytes, Device::CPU);
        LLAMA32_ASSERT(fread(host, weights_nbytes, 1, fin) == 1);
        ops::copy_host_to_cuda(host, t.w.emb_table, weights_nbytes);
        llama32_free(host, Device::CPU);
    }

    fclose(fin);
}


float* forward(Transformer& t, const int* tokens, const InferenceState& state)
{
    Timer timer{&metrics.inference_time_ms};

    ops::embed(tokens, t.w.emb_table, t.a.emb_acv, state);

    char* next_layer_inp = t.a.emb_acv;

    for (int i = 0; i < t.config.n_layers; i++) {
        ops::copy_tensors(next_layer_inp, t.a.layers[i].res_0_acv, state.n_ctx, state.d_embd, state);

        ops::rms_norm(next_layer_inp, t.w.layers[i].attn_norm, t.a.layers[i].attn_norm_acv, state);

        // ATTN
        // [n_ctx, n_emb], [d_out, d_embd]
        const int q_dim = state.n_heads * state.d_head;
        const int kv_dim = state.n_kv_heads * state.d_head;
        ops::matmul_2d(t.a.layers[i].attn_norm_acv, t.w.layers[i].q_proj, t.a.layers[i].q_proj_acv, state.d_embd, q_dim, state);
        ops::matmul_2d(t.a.layers[i].attn_norm_acv, t.w.layers[i].k_proj, t.a.layers[i].k_proj_acv, state.d_embd, kv_dim, state);
        ops::matmul_2d(t.a.layers[i].attn_norm_acv, t.w.layers[i].v_proj, t.a.layers[i].v_proj_acv, state.d_embd, kv_dim, state);
        ops::rotary_emb(t.a.layers[i].q_proj_acv, state.n_heads, state);
        ops::rotary_emb(t.a.layers[i].k_proj_acv, state.n_kv_heads, state);

        ops::qk(t.a.layers[i].q_proj_acv, t.a.layers[i].k_proj_acv, t.a.layers[i].qk_acv, state);
        ops::attn_mask_inplace(t.a.layers[i].qk_acv, state);
        ops::softmax_inplace(t.a.layers[i].qk_acv, state);
        ops::qkv(t.a.layers[i].qk_acv, t.a.layers[i].v_proj_acv, t.a.layers[i].qkv_acv, state);
        ops::matmul_2d(t.a.layers[i].qkv_acv, t.w.layers[i].o_proj, t.a.layers[i].o_proj_acv, state.d_embd, state.d_embd, state);

        ops::residual(t.a.layers[i].o_proj_acv, t.a.layers[i].res_0_acv, t.a.layers[i].res_1_acv, state);

        // MLP
        // self.w2(F.silu(self.w1(x)) * self.w3(x))
        // down(silu(gate(x)) * up(x))
        ops::rms_norm(t.a.layers[i].res_1_acv, t.w.layers[i].mlp_norm, t.a.layers[i].mlp_norm_acv, state);
        ops::matmul_2d(t.a.layers[i].mlp_norm_acv, t.w.layers[i].gate_proj, t.a.layers[i].mlp_gate_acv, state.d_embd, state.d_mlp, state);
        ops::matmul_2d(t.a.layers[i].mlp_norm_acv, t.w.layers[i].up_proj, t.a.layers[i].mlp_up_acv, state.d_embd, state.d_mlp, state);

        ops::silu_inplace(t.a.layers[i].mlp_gate_acv, state);
        ops::mul_inplace(t.a.layers[i].mlp_gate_acv, t.a.layers[i].mlp_up_acv, state);
        ops::matmul_2d(t.a.layers[i].mlp_gate_acv, t.w.layers[i].down_proj, t.a.layers[i].mlp_down_acv, state.d_mlp, state.d_embd, state);

        ops::residual(t.a.layers[i].res_1_acv, t.a.layers[i].mlp_down_acv, t.a.layers[i].res_1_acv, state);

        next_layer_inp = t.a.layers[i].res_1_acv;
    }

    ops::rms_norm(next_layer_inp, t.w.out_norm, t.a.out_norm_acv, state);
    
    ops::lm_head_proj(t.a.out_norm_acv, t.w.emb_table, t.a.logits_acv, state);

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

    float* cu_logits;
    if (t.device == Device::CUDA) {
        cu_logits = (float*)llama32_malloc(logits_size*sizeof(float), Device::CPU);
    }
    

    InferenceState state{t.device};

    const int n_pred_tokens = t.max_ctx - tokens.size();
    for (int i = 0; i < n_pred_tokens; i++) {
        state.n_ctx = tokens.size();
        state.start_pos = i == 0 ? 0 : tokens.size()-1;

        const float* logits = forward(t, tokens.data(), state);
        
        if (t.device == Device::CUDA) {
            ops::copy_cuda_to_host(logits, cu_logits, logits_size*sizeof(float));
            logits = cu_logits;
        }

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
        if (pred_token == tokenizer.eot_id) {
            break;
        }

        std::printf("%s", tokenizer.decode(pred_token));
        std::fflush(stdout);

        tokens.push_back(pred_token);
    }
    printf("\n");

    return tokens.size();
}

bool cuda_is_available() {
#if defined(__NVCC__)
    return true;
#else
    return false;
#endif
}

Device get_default_device() {
#if defined(__NVCC__)
    return Device::CUDA;
#else
    return Device::CPU;
#endif
}


static const char *usage_message = R"(
USAGE:
./llama32 [options] -p PROMPT  for a single prompt or
./llama32 [options] for a chat interface. 

Optional args. 
-f16 :     Use float-16 model (2.3GB). [default]
--dev DEVICE: The device to run inference on. Options are (cpu, cuda). Default is cuda if it is available.
--npred  N : Max context size. Minimum is 128 and max is 8192 [default=512]. Higher values consume more memory.
)";


int main(int argc, char const *argv[])
{
    Timer timer{&metrics.total_runtime_ms};

    const char* model_path = "models/llama32-1B.fp16.bin";
    int max_ctx = 512;
    std::string prompt = "";
    Device inference_device = get_default_device();

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
                return -1;
            }
        }
        else if (arg == "--dev") {
            if (i + 1 < argc) {
                std::string_view arg_device(argv[i + 1]);
                if (arg_device == "cuda") {
                    if (cuda_is_available()) {
                        inference_device = Device::CUDA;
                    } else {
                        fprintf(stdout, "Warning: cuda device is not available. Running inference on CPU.");
                    }
                } else if (arg_device == "cpu") {
                    inference_device = Device::CPU;
                } else {
                    fprintf(stderr, "error: invalid device argument `%s`. Allowed values are `cpu` or `cuda`.\n", arg_device.data());
                    return -1;
                }
                i += 1; // fast-forward
            } else {
                fprintf(stderr, "error: Device not provided.\n");
                return -1;
            }
        }
        else if (arg == "--npred") {
            if (argc <= i+1) {
                fprintf(stderr, "error: npred value is missing.\n");
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

    Transformer model{inference_device, max_ctx};

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

