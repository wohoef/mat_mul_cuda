#include <cstdio>
#include <iostream>
#include <cstdlib>

#define BM 64
#define BN 64
#define BK 8

#define TM 4
#define TN 4

struct Matrix {
    int rows, cols;
    float* data;
};

__global__ void matrixMult(Matrix A, Matrix B, Matrix C) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int col = blockIdx.x * BN + tx * TN;
    int row = blockIdx.y * BM + ty * TM;

    float c[TM][TN] = {0.0f};

    float a_reg[TM];
    float b_reg[TN];

    int numTiles = A.cols / BK;

    for (int t = 0; t < numTiles; t++) {
        int load_a_row = ty * 4 + tx / 4;
        int load_a_col = tx % 4;
        s_A[load_a_row][load_a_col] = A.data[(blockIdx.y * BM + load_a_row) * A.cols + (t * BK + load_a_col)];
        s_A[load_a_row][load_a_col + 4] = A.data[(blockIdx.y * BM + load_a_row) * A.cols + (t * BK + load_a_col + 4)];

        int load_b_row = ty / 4;
        int load_b_col = tx * 4 + ty % 4;
        s_B[load_b_row][load_b_col] = B.data[(t * BK + load_b_row) * B.cols + (blockIdx.x * BN + load_b_col)];
        s_B[load_b_row + 4][load_b_col] = B.data[(t * BK + load_b_row + 4) * B.cols + (blockIdx.x * BN + load_b_col)];

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                a_reg[i] = s_A[ty * TM + i][k];
            }
            for (int j = 0; j < TN; j++) {
                b_reg[j] = s_B[k][tx * TN + j];
            }

            // Multiply 16 times using the same registers
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            C.data[(row + i) * C.cols + (col + j)] = c[i][j];
        }
    }
}

int main(int argc, char** argv) {
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

    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 blocksPerGrid(
        N / BN,
        M / BM
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