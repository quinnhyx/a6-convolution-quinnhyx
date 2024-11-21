#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 5
#define WIDTH 10000
#define BLOCKSIZE 16

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch; // Memory pitch
    float* elements;
} Matrix;

// Function to allocate a matrix on host memory
Matrix AllocateMatrix(int height, int width, int init) {
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    if (init == 2) return M; // Structure only, no memory allocation

    M.elements = (float*)malloc(size * sizeof(float));

    for (unsigned int i = 0; i < size; i++) {
        M.elements[i] = (init == 0) ? 0.0f : (rand() / (float)RAND_MAX);
        if (rand() % 2) M.elements[i] = -M.elements[i];
    }

    return M;
}


// Constant memory for the mask
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

// CUDA Kernel for convolution
__global__ void ConvolutionKernel(Matrix d_N, Matrix d_P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary
    if (row < d_N.height && col < d_N.width) {
        float Pvalue = 0.0f;
        int maskRadius = MASK_WIDTH / 2;

        for (int i = -maskRadius; i <= maskRadius; i++) {
            for (int j = -maskRadius; j <= maskRadius; j++) {
                int neighborRow = row + i;
                int neighborCol = col + j;

                // Check if within bounds
                if (neighborRow >= 0 && neighborRow < d_N.height &&
                    neighborCol >= 0 && neighborCol < d_N.width) {
                    Pvalue += d_N.elements[neighborRow * d_N.pitch / sizeof(float) + neighborCol] * 
                              Mc[i + maskRadius][j + maskRadius];
                }
            }
        }

        // Write to output
        d_P.elements[row * d_P.pitch / sizeof(float) + col] = Pvalue;
    }
}

int main() {
    // Host matrices
    Matrix M = AllocateMatrix(MASK_WIDTH, MASK_WIDTH, 1);
    Matrix N = AllocateMatrix(WIDTH, WIDTH, 1);
    Matrix P = AllocateMatrix(WIDTH, WIDTH, 0);

    // Device matrices
    Matrix d_N = AllocateMatrix(WIDTH, WIDTH, 2);
    Matrix d_P = AllocateMatrix(WIDTH, WIDTH, 2);

    // Allocate device memory
    size_t pitch;
    cudaMallocPitch(&(d_N.elements), &pitch, WIDTH * sizeof(float), WIDTH);
    d_N.pitch = pitch;

    cudaMallocPitch(&(d_P.elements), &pitch, WIDTH * sizeof(float), WIDTH);
    d_P.pitch = pitch;

    // Copy host data to device
    cudaMemcpy2D(d_N.elements, d_N.pitch, N.elements, N.width * sizeof(float), 
                 N.width * sizeof(float), N.height, cudaMemcpyHostToDevice);

    // Copy mask to constant memory
    cudaMemcpyToSymbol(Mc, M.elements, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Kernel launch configuration
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, 
                 (WIDTH + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    double t0=get_clock();
    ConvolutionKernel<<<dimGrid, dimBlock>>>(d_N, d_P);
    cudaDeviceSynchronize();
    double t1=get_clock();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back to host
    cudaMemcpy2D(P.elements, P.width * sizeof(float), d_P.elements, d_P.pitch,
                 P.width * sizeof(float), P.height, cudaMemcpyDeviceToHost);

   printf("time per call: %f ns\n", (1000000000.0*(t1-t0)) );
   
    // Free device memory
    cudaFree(d_N.elements);
    cudaFree(d_P.elements);

    // Free host memory
    free(M.elements);
    free(N.elements);
    free(P.elements);

    return 0;
}
