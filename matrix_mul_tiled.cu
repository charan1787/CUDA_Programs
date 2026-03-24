#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matrixMulTiled(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int tiledColA = t * TILE + threadIdx.x;
        int tiledRowB = t * TILE + threadIdx.y;

        if (row < M && tiledColA < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tiledColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (tiledRowB < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 4, K = 4, N = 4;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE,
                   (M + TILE - 1) / TILE);

    matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Output matrix:\n";
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << h_C[row * N + col] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
