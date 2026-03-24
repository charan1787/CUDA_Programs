#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void reduceSum(const float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1024;
    size_t inputSize = N * sizeof(float);

    std::vector<float> h_input(N, 1.0f);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, inputSize);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t outputSize = numBlocks * sizeof(float);

    cudaMalloc((void**)&d_output, outputSize);

    cudaMemcpy(d_input, h_input.data(), inputSize, cudaMemcpyHostToDevice);

    reduceSum<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N);

    std::vector<float> h_output(numBlocks);
    cudaMemcpy(h_output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        finalSum += h_output[i];
    }

    std::cout << "Final sum = " << finalSum << "\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
