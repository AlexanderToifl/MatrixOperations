#include <stdio.h>
#include <stdlib.h>

#include "matrixOps.h"
#include "utils.h"

#define TILE_W 32

__global__ void squareMatrixAddKernel(float* d_A, float* d_B, float* d_C, int dim)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if( i < dim)
    {
        d_C[i] = d_A[i] + d_B[i];
    }

}

//matrix multiplication using shared memory
__global__ void squareMatrixMulKernel(float* d_A, float* d_B, float* d_C, int dim)
{
    __shared__ float As[TILE_W][TILE_W];
    __shared__ float Bs[TILE_W][TILE_W];
    
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    //printf("block_x = %d, block_y = %d\n",block_x, block_y); 
    
    int row = block_y * TILE_W + thread_y;
    int col = block_x * TILE_W + thread_x;
    
    float Cvalue = 0;
    
    
    for(int m = 0; m < ceil((float) dim/TILE_W); ++m)
    {
        //load from global into shared memory
         
        if(m * TILE_W + thread_x < dim && m * TILE_W + thread_y < dim) 
        {
            //printf("m * TILE_W + thread_x = %d, m * TILE_W + thread_y = %d\n",m * TILE_W + thread_x, m * TILE_W + thread_y);
            As[thread_y][thread_x] = d_A[row * dim + m * TILE_W + thread_x];
            Bs[thread_y][thread_x] = d_B[ (m * TILE_W + thread_y) * dim + col];
        }
        else
        {
             As[thread_y][thread_x] = 0;
             Bs[thread_y][thread_x] = 0;
        }
        __syncthreads();
        
        for( int k = 0; k < TILE_W; ++k )
        {
            Cvalue += As[thread_y][k] * Bs[k][thread_x];
        }
        __syncthreads();
    }

    //printf("row = %d, col = %d\n",row, col); 
    //printf("row * dim + col = %d\n",row * dim + col); 
    
    //write back to global memory
    if(row < dim && col < dim)
        d_C[row * dim + col] = Cvalue;
    

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


void squareMatrixMul(float* A, float* B, float* C, int dim)
{
    int size = dim *dim * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_A, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_B, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_C, size));
    
    
  
    dim3 dimGrid(ceil(dim/(float) TILE_W), ceil(dim/(float) TILE_W), 1);
    dim3 dimTiles(TILE_W, TILE_W, 1);
    
    printf("dimGrid = (%d, %d)\n", dimGrid.x, dimGrid.y);
    printf("dimTiles = (%d, %d)\n", dimTiles.x, dimTiles.y);
    
    squareMatrixMulKernel<<<dimGrid,dimTiles>>>(d_A, d_B, d_C,dim);
    cudaDeviceSynchronize();
    
    CUDA_CHECK_RETURN(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));
}
