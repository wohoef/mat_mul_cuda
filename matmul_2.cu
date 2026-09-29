#include <cstdio>
#include <iostream>
#include <cstdlib>

#define TILE_SIZE 16

struct Matrix {
    int rows, cols;
    float* data;
};

__global__ void matrixMult(Matrix A, Matrix B, Matrix C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0f;

    int numTiles = (A.cols + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0)
}

int main(int argc, char** argv) {
    int size = std::atoi(argv[1]);

    int M = size; // A rows
    int K = size; // A cols and B rows
    int N = size; // B cols

    Matrix h_A = {M, K, new float[M * K]};
    Matrix h_B = {K, N, new float[K * N]};
    Matrix h_C = {M, N, new float[M * N]};

    for (int i = 0; i < M * K; i++) h_A.data[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B.data[i] = 1.0f;

    Matrix d_A = {M, K, nullptr};
    Matrix d_B = {K, N, nullptr};
    Matrix d_C = {M, N, nullptr};

    cudaMalloc(&d_A.data, M * K * sizeof(float));
    cudaMalloc(&d_B.data, K * N * sizeof(float));
    cudaMalloc(&d_C.data, M * N * sizeof(float));

    cudaMemcpy(d_A.data, h_A.data, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.data, h_B.data, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms << std::endl;


    cudaFree(d_A.data);
    cudaFree(d_B.data);
    cudaFree(d_C.data);

    delete[] h_A.data;
    delete[] h_B.data;
    delete[] h_C.data;
}