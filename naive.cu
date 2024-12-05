#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCKSIZE 16
#define Mask_Width 5
#define WIDTH 10000000

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_width, int Width){
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        float Pvalue = 0;
        int N_start_point = i - (Mask_Width/2);
        for (int j = 0; j < Mask_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
               Pvalue += N[N_start_point + j]*M[j];
            }
            __syncthreads();
        }
        P[i] = Pvalue;
}

int main(){
    float *N, *M, *P;
    cudaMallocManaged(&N, sizeof(float)*WIDTH);
    cudaMallocManaged(&M, sizeof(float)*Mask_Width);
    cudaMallocManaged(&P, sizeof(float)*WIDTH);

    for(int i=0;i<WIDTH;i++){
         N[i] = 1;
    }
    for(int i=0;i<Mask_Width;i++){
         M[i] = 1;
    }
    
    double t0=get_clock();
    convolution_1D_basic_kernel<<<(WIDTH + BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(N, M, P, Mask_Width, WIDTH);
    cudaDeviceSynchronize();
    double t1 = get_clock();

    printf("time per call: %f ns\n", (1000000000.0*(t1-t0)));

    cudaFree(N);
    cudaFree(M);
    cudaFree(P);
    return 0;
}

