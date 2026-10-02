#include <cstdio>
#include <iostream>
#include <cstdlib>

struct Matrix {
    int rows, cols;
    float* data;
};

__global__ void matrixMult(Matrix A, Matrix B, Matrix C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < A.rows && col < B.cols) {
        float sum = 0.0f;
        for (int k=0; k < A.cols; k++) {
            sum += A.data[row * A.cols + k] * B.data[k * B.cols + col];
        }
        C.data[row * C.cols + col] = sum;
    }
}

int main(int argc, char** argv) {
    std::cout << sizeof(float) << std::endl;
    int size = std::atoi(argv[1]);

    int M = size; // A rows
    int K = size; // A cols and B rows
    int N = size; // B cols

    // Host matrices
    Matrix h_A = {M, K, new float[M * K]};
    Matrix h_B = {K, N, new float[K * N]};
    Matrix h_C = {M, N, new float[M * N]};

    for (int i = 0; i < M * K; i++) h_A.data[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B.data[i] = 1.0f;

    // Device matrices
    Matrix d_A = {M, K, nullptr};
    Matrix d_B = {K, N, nullptr};
    Matrix d_C = {M, N, nullptr};

    cudaMalloc(&d_A.data, M * K * sizeof(float));
    cudaMalloc(&d_B.data, K * N * sizeof(float));
    cudaMalloc(&d_C.data, M * N * sizeof(float));

    cudaMemcpy(d_A.data, h_A.data, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.data, h_B.data, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Warm up run
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Actual run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    // Check for errors
    cudaError_t syncErr = cudaEventSynchronize(stop);
    if (syncErr != cudaSuccess) {
        std::cout << "Sync Error: " << cudaGetErrorString(syncErr) << std::endl;
        return;
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms << std::endl;

    // Cleanup
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    delete[] h_A.data;
    delete[] h_B.data;
    delete[] h_C.data;
}