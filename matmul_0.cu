#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cuBLAS is loaded at runtime with dlopen, so this file builds with the same
// `nvcc -arch=native -o matmul_0 matmul_0.cu` line as the other kernels.
// No -lcublas needed, so the benchmark script doesn't have to change.

static void* loadCublas() {
    const char* names[] = {
        "libcublas.so",
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "/usr/local/cuda/lib64/libcublas.so",
    };
    for (const char* n : names) {
        if (void* h = dlopen(n, RTLD_NOW | RTLD_GLOBAL)) return h;
    }
    std::cerr << "Could not load libcublas: " << dlerror() << std::endl;
    std::exit(1);
}

template <typename T>
static T sym(void* lib, const char* name) {
    void* p = dlsym(lib, name);
    if (!p) {
        std::cerr << "Missing cuBLAS symbol: " << name << std::endl;
        std::exit(1);
    }
    return reinterpret_cast<T>(p);
}

int main(int argc, char** argv) {
    int size = std::atoi(argv[1]);

    int M = size; // A rows
    int K = size; // A cols and B rows
    int N = size; // B cols

    void* lib = loadCublas();
    auto create  = sym<decltype(&cublasCreate_v2)>(lib, "cublasCreate_v2");
    auto destroy = sym<decltype(&cublasDestroy_v2)>(lib, "cublasDestroy_v2");
    auto sgemm   = sym<decltype(&cublasSgemm_v2)>(lib, "cublasSgemm_v2");

    // Host matrices
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    if (create(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed" << std::endl;
        return 1;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major. Row-major C = A * B is the same memory as
    // column-major C^T = B^T * A^T, so pass B first and swap M/N.
    auto run = [&]() {
        sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              d_B, N,
              d_A, K,
              &beta,
              d_C, N);
    };

    // Warm up run (also lets cuBLAS pick and load its kernel)
    run();

    // Actual run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    run();
    cudaEventRecord(stop);

    cudaError_t syncErr = cudaEventSynchronize(stop);
    if (syncErr != cudaSuccess) {
        std::cout << "Sync Error: " << cudaGetErrorString(syncErr) << std::endl;
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms << std::endl;

    // Optional: dump A, B and C to a file so test.py can check the result
    if (argc > 2) {
        float* h_C = new float[M * N];
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        FILE* f = std::fopen(argv[2], "wb");
        std::fwrite(h_A, sizeof(float), M * K, f);
        std::fwrite(h_B, sizeof(float), K * N, f);
        std::fwrite(h_C, sizeof(float), M * N, f);
        std::fclose(f);
        delete[] h_C;
    }

    // Cleanup
    destroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
}