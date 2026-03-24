#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void blur(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;

    // 3x3 neighborhood
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            // boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }

    output[y * width + x] = sum / count;
}

int main() {
    int width = 5;
    int height = 5;
    int N = width * height;
    size_t size = N * sizeof(float);

    // Simple 5x5 input image
    std::vector<float> h_input = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    std::vector<float> h_output(N, 0.0f);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    blur<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Input image:\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << h_input[y * width + x] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nBlurred output:\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << h_output[y * width + x] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}