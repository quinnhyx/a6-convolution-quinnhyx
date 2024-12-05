#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define TILE_WIDTH 256  // Define the block size
#define RADIUS 2        // Define the radius of the convolution mask
#define WIDTH 10000000
#define MASK_WIDTH 5

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// Kernel function
__global__ void convolution_1D_tiled_cache_kernel(float *N, float *P, const float *M, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global index
    int radius = Mask_Width / 2;  // Half-width of the mask

    // Shared memory size accommodates TILE_WIDTH + halo regions
    __shared__ float N_ds[TILE_WIDTH + 2 * RADIUS];

    // Load shared memory with halo regions
    int shared_idx = threadIdx.x + radius;
    N_ds[shared_idx] = (i < Width) ? N[i] : 0;  // Main element

    // Load halo regions
    if (threadIdx.x < radius) {
        // Left halo
        N_ds[threadIdx.x] = (i - radius >= 0) ? N[i - radius] : 0;
        // Right halo
        N_ds[shared_idx + blockDim.x] = (i + blockDim.x < Width) ? N[i + blockDim.x] : 0;
    }

    __syncthreads();

    // Compute convolution
    float Pvalue = 0.0f;
    for (int j = 0; j < Mask_Width; j++) {
        Pvalue += N_ds[shared_idx - radius + j] * M[j];
    }

    // Write result to global memory
    if (i < Width) {
        P[i] = Pvalue;
    }
}

// Utility to check CUDA errors
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Main function
int main() {
    const int Width = WIDTH;        // Input array size
    const int Mask_Width = MASK_WIDTH;    // Convolution mask width

    // Allocate and initialize host memory
    float *h_N = (float *)malloc(Width * sizeof(float));
    float *h_P = (float *)malloc(Width * sizeof(float));
    float *h_M = (float *)malloc(Mask_Width * sizeof(float));

    for (int i = 0; i < Width; i++) {
        h_N[i] = 1;  // Example input
    }
    for (int i = 0; i < Mask_Width; i++) {
        h_M[i] = 1;  // Simple average mask
    }

    // Allocate device memory
    float *d_N, *d_P, *d_M;
    checkCuda(cudaMalloc((void **)&d_N, Width * sizeof(float)), "cudaMalloc d_N");
    checkCuda(cudaMalloc((void **)&d_P, Width * sizeof(float)), "cudaMalloc d_P");
    checkCuda(cudaMalloc((void **)&d_M, Mask_Width * sizeof(float)), "cudaMalloc d_M");

    // Copy data to device
    checkCuda(cudaMemcpy(d_N, h_N, Width * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_N to d_N");
    checkCuda(cudaMemcpy(d_M, h_M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_M to d_M");

    // Kernel launch configuration
    int threadsPerBlock = TILE_WIDTH;
    int blocksPerGrid = (Width + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    double t0 = get_clock();
    convolution_1D_tiled_cache_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_P, d_M, Mask_Width, Width);
    checkCuda(cudaGetLastError(), "Kernel launch");
    cudaDeviceSynchronize();
    double t1 = get_clock();
    printf("time per call: %f ns\n", (1000000000.0*(t1-t0)));

    // Copy result back to host
    checkCuda(cudaMemcpy(h_P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_P to h_P");

    // Free device and host memory
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_M);
    free(h_N);
    free(h_P);
    free(h_M);

    return 0;
}
