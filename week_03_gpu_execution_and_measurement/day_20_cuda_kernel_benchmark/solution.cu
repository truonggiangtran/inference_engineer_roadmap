// Day 20 solution: validate and benchmark a SAXPY kernel with CUDA events.
// Build: nvcc -O3 -arch=sm_120 -o saxpy solution.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                              \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

constexpr int kBlock = 256;
constexpr int kWarmup = 10;
constexpr int kRepeats = 50;

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

struct Stats {
    float median, p10, p90;
};

Stats summarize(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    auto at = [&](double q) { return samples[static_cast<size_t>(q * (samples.size() - 1))]; };
    return {at(0.5), at(0.1), at(0.9)};
}

bool validate(int n, float a, const std::vector<float>& x, const std::vector<float>& y0,
              const std::vector<float>& result, double* max_rel_error) {
    const double tolerance = 1e-6;  // one fused multiply-add in FP32
    *max_rel_error = 0.0;
    for (int i = 0; i < n; ++i) {
        double expected = static_cast<double>(a) * x[i] + y0[i];
        double rel = std::fabs(result[i] - expected) / std::max(std::fabs(expected), 1e-12);
        *max_rel_error = std::max(*max_rel_error, rel);
    }
    return *max_rel_error <= tolerance;
}

bool run(int n, std::FILE* csv) {
    const float a = 2.0f;
    std::vector<float> x(n), y0(n), result(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 1000) * 0.001f;
        y0[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice));

    const int grid = (n + kBlock - 1) / kBlock;
    saxpy<<<grid, kBlock>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    double max_rel_error = 0.0;
    bool ok = validate(n, a, x, y0, result, &max_rel_error);
    std::printf("n=%d validation %s (max rel error %.3e)\n", n, ok ? "PASSED" : "FAILED", max_rel_error);

    std::vector<float> samples;
    for (int i = 0; i < kWarmup; ++i) saxpy<<<grid, kBlock>>>(n, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < kRepeats; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        saxpy<<<grid, kBlock>>>(n, a, d_x, d_y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        samples.push_back(ms);
    }

    if (samples.empty()) {
        std::printf("n=%d timing not implemented yet (TODO)\n", n);
    } else {
        Stats s = summarize(samples);
        double gbps = 3.0 * bytes / (s.median / 1e3) / 1e9;
        std::printf("n=%d median %.4f ms (p10 %.4f, p90 %.4f) effective %.1f GB/s\n", n, s.median, s.p10, s.p90, gbps);
        if (csv) std::fprintf(csv, "%d,%.6f,%.6f,%.6f,%.2f,%s\n", n, s.median, s.p10, s.p90, gbps, ok ? "pass" : "fail");
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return ok && !samples.empty();
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    std::FILE* csv = std::fopen("results/saxpy.csv", "w");
    if (!csv) std::printf("note: create a results/ directory to save the CSV\n");
    if (csv) std::fprintf(csv, "n,median_ms,p10_ms,p90_ms,effective_gbps,validation\n");

    bool ok = true;
    for (int n : {1 << 16, 1 << 26}) ok = run(n, csv) && ok;
    if (csv) std::fclose(csv);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
