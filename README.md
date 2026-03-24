#  CUDA Programs

This repo contains 5 CUDA programs:

1. `vector_add.cu`
2. `saxpy.cu`
3. `matrix_mul_naive.cu`
4. `matrix_mul_tiled.cu`
5. `reduction_sum.cu`

These are intentionally kept simple and beginner-friendly.

### 1. Vector Add
- first CUDA kernel
- one thread per element
- host to device copy
- bounds check

### 2. SAXPY
- element-wise parallel computation
- same indexing idea as vector add
- simple arithmetic kernel

### 3. Naive Matrix Multiplication
- 2D blocks and grids
- row and column mapping
- one thread per output element

### 4. Tiled Matrix Multiplication
- shared memory
- tiling
- `__syncthreads()`
- reducing global memory access

### 5. Reduction Sum
- tree-style reduction
- shared memory
- synchronization
- block-level partial sums

## Compile

Use `nvcc`:

```bash
nvcc vector_add.cu -o vector_add
nvcc saxpy.cu -o saxpy
nvcc matrix_mul_naive.cu -o matrix_mul_naive
nvcc matrix_mul_tiled.cu -o matrix_mul_tiled
nvcc reduction_sum.cu -o reduction_sum
```

## Run

```bash
./vector_add
./saxpy
./matrix_mul_naive
./matrix_mul_tiled
./reduction_sum
```
