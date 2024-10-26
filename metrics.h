#pragma once

#include <chrono>
#include <cstdio>


struct Metrics {
    int64_t matmul_ms = 0;
    int64_t non_matmul_ms = 0;
    int64_t load_time_ms = 0;
    int64_t inference_time_ms = 0;
    int64_t total_runtime_ms = 0;
    int64_t sample_time_ms = 0;
    int64_t total_mem_bytes = 0;
};

static Metrics metrics;


void print_metrics(const Metrics& m, int n_tokens/*number of processed tokens.*/)
{
    std::printf("\n-------------------------------\n");
    std::printf(" METRICS\n");
    std::printf("-------------------------------\n");
    std::printf(" Sample time         : %5ldms\n", m.sample_time_ms);
    std::printf(" Load time           : %5ldms\n", m.load_time_ms);
    std::printf(" Inference [per tok] : %5ldms\n", m.inference_time_ms/n_tokens);
    std::printf(" Inference [total]   : %5ldms\n", m.inference_time_ms);
    std::printf(" Total runtime       : %5ldms\n", m.total_runtime_ms);
    std::printf("-------------------------------\n");
    std::printf(" Mem usage [total]   : %5ldMB\n", m.total_mem_bytes/1000000);
    // std::cout << " " << "Mem usage [model]   : " << std::setw(4) << weights_mem/1000000 << "MB\n";
    // std::cout << " " << "Mem usage [actvs]   : " << std::setw(4) << acv_mem/1000000 << "MB\n";
    std::printf("-------------------------------\n");
    std::printf(" Matmul   [per tok]  : %5ldms\n", m.matmul_ms/n_tokens);
    std::printf(" NonMatmul [per tok] : %5ldms\n", m.non_matmul_ms/n_tokens);
    std::printf("-------------------------------\n\n");
}



class Timer {
public:
    Timer(int64_t* duration_ptr)
        : m_duration_ptr{duration_ptr},
          m_start_time{std::chrono::high_resolution_clock::now()}
    { 
    }
    ~Timer() { stop(); }

    void stop() {
        if (m_stopped) {
            return;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_start_time).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *m_duration_ptr += duration;
        m_stopped = true;
    }

private:
    int64_t* m_duration_ptr;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    bool m_stopped = false;
};
