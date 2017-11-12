#include <stdio.h>
#include <stdlib.h>

#include "vectorOps.h"
#include "utils.h"

#define MAX_MASK_WIDTH 16
#define TILE_SIZE 64

 __constant__ float d_mask[MAX_MASK_WIDTH];

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if( i < n )
    {
        C[i] = A[i] + B[i];
    }

}

__global__ void convolutionKernel(float* in, float* out,  int in_size, int mask_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tile[TILE_SIZE + MAX_MASK_WIDTH - 1];
    
    
    int n = mask_size / 2; //assuming mask_size is odd
    
    int halo_idl = (blockIdx.x - 1) * blockDim.x + threadIdx.x; //left halo index
    if( threadIdx.x >= blockDim.x - n)
    {
        tile[threadIdx.x - (blockDim.x - n)] = (halo_idl < 0) ? 0 : in[halo_idl];
    }
    
    tile[n + threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
    
    int halo_idr = (blockIdx.x + 1) * blockDim.x + threadIdx.x; //right halo index
    if( threadIdx.x <  n)
    {
        tile[n + blockDim.x + threadIdx.x] = (halo_idr >= in_size) ? 0 : in[halo_idr];
    }
    
    __syncthreads();
    
    float outvalue = 0;
    
    for( int j = 0; j < mask_size; ++j )
    {
        outvalue += tile[threadIdx.x + j] * d_mask[j];
    }
    
    out[i] = outvalue;
    
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

void convolution(float* in, float* out, float* mask, int in_size, int mask_size)
{
    float *d_in, *d_out;

    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_in, in_size*sizeof(float) ) );
    CUDA_CHECK_RETURN(cudaMemcpy(d_in, in, in_size*sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_out, in_size*sizeof(float)));
 
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mask, mask, mask_size * sizeof(float) ) );
    
    convolutionKernel<<<ceil(in_size/(float) TILE_SIZE), TILE_SIZE>>>(d_in, d_out, in_size, mask_size);
    
    CUDA_CHECK_RETURN(cudaMemcpy(out, d_out, in_size*sizeof(float), cudaMemcpyDeviceToHost));
    
    
}
