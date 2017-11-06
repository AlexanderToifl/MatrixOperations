#include <stdio.h>
#include <stdlib.h>

#include "matrixOps.h"
#include "utils.h"

#define TILE_W 32


void printMat(float *A, int nrows, int ncols)
{
    printf("[\t");
    for(int iy = 0; iy < nrows; ++iy)
    {
        for(int ix = 0; ix < ncols; ++ix)
        {
            printf("%f ", A[ncols * iy + ix]);
        }
        printf("\n\t");
    }
    
    printf("\n]\n");
}    

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

//matrix multiplication using shared memory
// dim_c .... common dimension because A_ncols = B_nrows
__global__ void matrixMulKernel(float* d_A, float* d_B, float* d_C,int A_nrows, int dim_c, int B_ncols)
{
    __shared__ float As[TILE_W][TILE_W];
    __shared__ float Bs[TILE_W][TILE_W];
    
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    
    int row = block_y * TILE_W + thread_y;
    int col = block_x * TILE_W + thread_x;
    
    float Cvalue = 0;
    
    for(int m = 0; m < ceil((float) dim_c/TILE_W); ++m)
    {
        //initialize matrix tiles in shared memory to 0
        As[thread_y][thread_x] = 0;
        Bs[thread_y][thread_x] = 0;
        
        //load from global into shared memory
         
        if(m * TILE_W + thread_x < dim_c && row < A_nrows) 
            As[thread_y][thread_x] = d_A[row * dim_c + m * TILE_W + thread_x];

        if( m * TILE_W + thread_y < dim_c && col < B_ncols)  
            Bs[thread_y][thread_x] = d_B[ (m * TILE_W + thread_y) * B_ncols + col];
       
        __syncthreads();
        
        for( int k = 0; k < TILE_W; ++k )
        {
            Cvalue += As[thread_y][k] * Bs[k][thread_x];
        }
        __syncthreads();
    }

    //write back to global memory
    if(row < A_nrows && col < B_ncols)
        d_C[row * B_ncols + col] = Cvalue;

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

void matrixMul(float* A, float* B, float* C, int A_nrows, int A_ncols, int B_nrows, int B_ncols)
{
    if( A_ncols != B_nrows)
    {
        printf("Error in matrixMul: dimensions do not match. A = (%d, %d), B = (%d, %d)\n", A_nrows, A_ncols, B_nrows, B_ncols);
        return;
    }
    
    
    int Asize = A_nrows * A_ncols * sizeof(float);
    int Bsize = B_nrows * B_ncols * sizeof(float);
    int Csize = A_nrows * B_ncols * sizeof(float);
    
    int dim_c = A_ncols;
    float *d_A, *d_B, *d_C;
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_A, Asize));
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A, Asize, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_B, Bsize));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, B, Bsize, cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_C, Csize));
    
    //a square grid of tiles is used, x and y dimensions are chosen to be large enough to cover all elements in A and B
    int square_tiles_dim =  ceil(max( max(A_nrows, B_nrows), max(A_ncols, B_ncols ) ) /( (float) TILE_W) );
    
    dim3 dimGrid(square_tiles_dim, square_tiles_dim, 1);
    dim3 dimTiles(TILE_W, TILE_W, 1);
    
    printf("dimGrid = (%d, %d)\n", dimGrid.x, dimGrid.y);
    printf("dimTiles = (%d, %d)\n", dimTiles.x, dimTiles.y);
    
    matrixMulKernel<<<dimGrid,dimTiles>>>(d_A, d_B, d_C, A_nrows, dim_c, B_ncols);
    cudaDeviceSynchronize();
    
    CUDA_CHECK_RETURN(cudaMemcpy(C, d_C, Csize, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));
}
