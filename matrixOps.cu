#include <stdio.h>
#include <stdlib.h>

#include "matrixOps.h"
#include "utils.h"



__global__ void squareMatrixAddKernel(float* A, float* B, float* C, int dim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if( i < dim)
    {
        C[i] = A[i] + B[i];
    }

}

void squareMatrixAdd(float* A, float* B, float* C, int dim)
{
    unsigned int THREADS_PER_BLOCK = 256;
    
    int size = dim *dim * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_A, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_B, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_C, size));
    
    squareMatrixAddKernel<<<ceil(dim*dim/(float) THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A, d_B, d_C,dim);
    
    CUDA_CHECK_RETURN(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));
    
}

