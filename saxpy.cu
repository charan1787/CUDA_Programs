#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void saxpy(float a, const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] = a * X[i] + Y[i];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    float a = 2.0f;

    std::vector<float> h_X(N, 1.0f);
    std::vector<float> h_Y(N, 3.0f);

    float *d_X, *d_Y;

    cudaMalloc((void**)&d_X, size);
    cudaMalloc((void**)&d_Y, size);

    cudaMemcpy(d_X, h_X.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    saxpy<<<numBlocks, threadsPerBlock>>>(a, d_X, d_Y, N);

    cudaMemcpy(h_Y.data(), d_Y, size, cudaMemcpyDeviceToHost);

    std::cout << "First 5 results:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
