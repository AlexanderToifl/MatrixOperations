#include <stdio.h>
#include <stdlib.h>

#include "vectorOps.h"
#include "utils.h"



__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if( i < n )
    {
        C[i] = A[i] + B[i];
    }

}

void vecAdd(float* A, float* B, float* C, int n)
{
    unsigned int THREADS_PER_BLOCK = 256;
    
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_A, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_B, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_C, size));
    
    vecAddKernel<<<ceil(n/(float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A, d_B, d_C,n);
    
    CUDA_CHECK_RETURN(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));
    
}

