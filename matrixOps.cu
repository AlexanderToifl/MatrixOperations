#include <stdio.h>
#include <stdlib.h>

#include "matrixOps.h"
#include "utils.h"

#define DEBUG 0

#define TILE_W 16
#define MAX_MASK_WIDTH 17

 __constant__ float d_mask[MAX_MASK_WIDTH * MAX_MASK_WIDTH];




__device__ int checkIndicesMat(int ix, int iy, int ncols, int nrows)
{
    if(ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        return 1;
    else 
        return 0;
}



__device__ __host__ void printMat(float *A, int nrows, int ncols)
{
    printf("[\t");
    for(int iy = 0; iy < nrows; ++iy)
    {
        for(int ix = 0; ix < ncols; ++ix)
        {
            printf("%.2f ", A[ncols * iy + ix]);
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
//assuming sqaure mask and tiles
__global__ void convolution2DKernel(float* in, float* out, int in_nrows, int in_ncols, int mask_dim)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float tile[TILE_W + MAX_MASK_WIDTH - 1][TILE_W + MAX_MASK_WIDTH - 1];
    
    int n = (mask_dim  - 1 ) / 2; //assuming mask_dim is odd
    

    //coordinates in global memory that are associated with this thread
    int gx = bx * TILE_W + tx - n;
    int gy = by * TILE_W + ty - n;
    
    //inner indices [xi_l, xi_r) = [yi_l, yi_r)
    int di_l = n;
    int di_r = di_l + TILE_W;
    
    //halo indices
    int hx = 0; 
    int hy = 0;
    
    float ghostValue = 0;
    
#if DEBUG
    int test = 0;
#endif    
    
    hx = bx * TILE_W + tx - n;
    hy = by * TILE_W + ty - n;
    
    if (checkIndicesMat(hx, hy, in_ncols, in_nrows) == 1)
        tile[ty][tx] = in[hy * in_ncols + hx];
    else
        tile[ty][tx] = ghostValue;
    
     __syncthreads();
    
    if(hx >= in_ncols || hy >= in_nrows)
        return;
    
    
    if(tx >= di_l && tx < di_r && ty >= di_l && ty < di_r)
    {
        
        float outvalue = 0;
    
        for( int u = 0; u < mask_dim; ++u )
        {
            for ( int v = 0; v < mask_dim; ++v)
            {
                outvalue += tile[ty - n + v][tx - n + u] * d_mask[v * mask_dim + u];
            }
        }
        
        out[hy * in_ncols + hx] = outvalue;
    }
    
#if DEBUG
    __syncthreads();
    if(bx * blockDim.x + tx == 4 && by * blockDim.y + ty  ==1)
    {
        printMat( (float*) tile, TILE_W + MAX_MASK_WIDTH - 1, TILE_W + MAX_MASK_WIDTH - 1); 
        printf("hx = %d, hy = %d\n", hx, hy);
    }
#endif    
    
    
}

/*__global__ void convolution2DKernel(float* in, float* out, int in_nrows, int in_ncols, int mask_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float tile[TILE_W + MAX_MASK_WIDTH - 1][TILE_W + MAX_MASK_WIDTH - 1];
    
    if(i >= in_ncols || j >= in_nrows)
        return;
    
    
    int n = (mask_dim -1) / 2; //assuming mask_dim is odd

    //printf("n = %d, blockDim.x = %d, blockDim.y = %d\n", n, blockDim.x, blockDim.y);
    
    int halo_idl = (blockIdx.x - 1) * blockDim.x + threadIdx.x; //left halo index
    int halo_idr = (blockIdx.x + 1) * blockDim.x + threadIdx.x; //right halo index
    int halo_idt = (blockIdx.y - 1) * blockDim.y + threadIdx.y; //top halo index
    int halo_idb = (blockIdx.y + 1) * blockDim.y + threadIdx.y; //bottom halo index
  
    float ghostValue = 0.0;
    int test = 0;
    int test2 = 0;
  
    //tile[threadIdx.y + n][threadIdx.x + n] = in[j * in_ncols + i];
    
    if( threadIdx.x >= blockDim.x - n)
    {
        if(threadIdx.y >= blockDim.y - n)
        {
            tile[threadIdx.y + n - blockDim.y][threadIdx.x + n - blockDim.x] = (halo_idt < 0 || halo_idl < 0) ? ghostValue : in[halo_idt * in_ncols + halo_idl];
        }
        else if(threadIdx.y <  n)
        {
            tile[threadIdx.y + n + blockDim.y][threadIdx.x + n - blockDim.x] = (halo_idb >= in_nrows || halo_idl < 0) ? ghostValue : in[halo_idb * in_ncols + halo_idl];
        }
        else
        {
            tile[threadIdx.y + n][threadIdx.x + n - blockDim.x] = (halo_idl < 0) ? ghostValue : in[j * in_ncols + halo_idl];
        }
        
    }
    else if( threadIdx.x <  n)
    {
        if(threadIdx.y >= blockDim.y - n)
        {
            tile[threadIdx.y + n - blockDim.y][threadIdx.x + n + blockDim.x] = (halo_idr >= in_ncols || halo_idt < 0) ? ghostValue : in[halo_idt * in_ncols + halo_idr];
        }
        else if(threadIdx.y <  n )
        {
            test = 1;
            tile[threadIdx.y + n + blockDim.y][threadIdx.x + n + blockDim.x] = (halo_idr >= in_ncols || halo_idb >= in_nrows) ? ghostValue : in[halo_idb * in_ncols + halo_idr];
        }
        else
        {
           
            tile[threadIdx.y + n][threadIdx.x + n + blockDim.x] =  (halo_idr >= in_ncols) ? ghostValue : in[j * in_ncols + halo_idr];
        }
    }
    else
    {
        if(threadIdx.y >= blockDim.y - n)
        {
            tile[threadIdx.y + n - blockDim.y][threadIdx.x + n] = (halo_idt < 0) ? ghostValue : in[halo_idt * in_ncols + i];
        }
        else if( threadIdx.y < n)
        {
            tile[threadIdx.y + n + blockDim.y][threadIdx.x + n] = (halo_idb >= in_nrows) ? ghostValue : in[halo_idb * in_ncols + i];
        }
    }
    
    __syncthreads();
    
    float outvalue = 0;
    
    for( int u = 0; u < mask_dim; ++u )
    {
        for ( int v = 0; v < mask_dim; ++v)
        {
            outvalue += tile[threadIdx.y + v][threadIdx.x + u] * d_mask[v * mask_dim + u];
        }
    }
    
    __syncthreads();
    
    out[j * in_ncols + i] = outvalue;
    
    if(i == 0 && j == 0)
    {
        printMat( (float*) tile, TILE_W + MAX_MASK_WIDTH - 1, TILE_W + MAX_MASK_WIDTH - 1); 
        printf("halo_idl = %d, halo_idr = %d, halo_idb = %d, halo_idt = %d\n", halo_idl, halo_idr, halo_idb, halo_idt);
        printf("test = %d, test2 = %d\n", test,test2);
    }
}
*/


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



void convolution2D(float* in, float* out, float* mask, int in_nrows, int in_ncols, int mask_dim)
{
    float *d_in, *d_out;

    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_in, in_nrows * in_ncols * sizeof(float) ) );
    CUDA_CHECK_RETURN(cudaMemcpy(d_in, in, in_nrows * in_ncols * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_out, in_nrows * in_ncols * sizeof(float)));
 
 
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_mask, mask, mask_dim * mask_dim * sizeof(float) ) );
    
    int in_dim = max(in_ncols, in_nrows);
    
    dim3 dimGrid(ceil(in_dim/(float) TILE_W), ceil(in_dim/(float) TILE_W), 1);
    dim3 dimTiles(TILE_W + mask_dim - 1, TILE_W + mask_dim - 1, 1);
    
    printf("mask_dim = %d\n", mask_dim);
    printf("dimGrid = (%d, %d)\n", dimGrid.x, dimGrid.y);
    printf("dimTiles = (%d, %d)\n", dimTiles.x, dimTiles.y);
    
    convolution2DKernel<<<dimGrid, dimTiles>>>(d_in, d_out, in_nrows, in_ncols, mask_dim);
    
    CUDA_CHECK_RETURN(cudaMemcpy(out, d_out,in_nrows * in_ncols * sizeof(float), cudaMemcpyDeviceToHost));
    
    
}
