//matrix_main.cu


#include <stdio.h>
#include <stdlib.h>
#include <time.h>  


#include "utils.h"
#include "vectorOps.h"
#include "matrixOps.h"

#define SQUARE 0

int main( int argc, char **argv )
{
    int seed = time(NULL);
    srand(seed);
    
#if SQUARE    
    if(argc < 2)
    {
        printf("1 argument required: integer number (vector size)\n");
        return 0;
    }

    
    
    int n = atoi(argv[1]);
    printf("n = %d\n", n);
    
    float* A = (float*) malloc(n*sizeof(float));
    float* B = (float*) malloc(n*sizeof(float));
    float* C = (float*) malloc(n*sizeof(float));
    
    int i=0;
    
    for(; i < n; ++i)
    {
        A[i] = 1;//rand() % 10 ;
        B[i] = rand() % 10 ;
        
    }
    
    vecAdd(A, B, C, n); 
    printf("vecAdd: C[1] = %f\n",C[1]);
    
    float mask[] = {1, 3, 5, 3, 1};
    
    convolution(A, B, mask, n, 5);
    
    printf("convolution: B[0:6] = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",B[0], B[1], B[2], B[3], B[4], B[5], B[6]);

    
    freeMemory( (void**) &A);
    freeMemory( (void**) &B);
    freeMemory( (void**) &C);
    return 0;
    
    
    float* MA = (float*) malloc(n*n*sizeof(float));
    float* MB = (float*) malloc(n*n*sizeof(float));
    float* MC = (float*) malloc(n*n*sizeof(float));
    
    for( i = 0; i < n*n; ++i)
    {
        MA[i] = 1;//rand() % 5;
        MB[i] = 1;//rand() % 5;
    }
    
    
    
    //squareMatrixAdd(MA, MB, MC, n);
    
    squareMatrixMul(MA, MB, MC, n);
    printf("MC[2,3] = %f\n", MC[2*n + 3]);
    

    
    freeMemory( (void**) &MA);
    freeMemory( (void**) &MB);
    freeMemory( (void**) &MC);
 
#else
    if(argc < 5)
    {
        printf("4 argument required: integer numbers.\n\tMatrix A dimensions (a_nrows, a_ncols)\n\tMatrix B dimensions (b_nrows, b_ncols)\n");
        return 0;
    }
    
    int a_nrows = atoi(argv[1]);
    int a_ncols = atoi(argv[2]);
    int asize = a_nrows*a_ncols;
    
    int b_nrows = atoi(argv[3]);
    int b_ncols = atoi(argv[4]);
    int bsize = b_nrows*b_ncols;
    
    int csize = a_nrows*b_ncols;
    
    printf("A = (%d, %d), B = (%d, %d)\n", a_nrows, a_ncols, b_nrows, b_ncols);
    
    float* A = (float*) malloc(asize*sizeof(float));
    float* B = (float*) malloc(bsize*sizeof(float));
    float* C = (float*) malloc(csize*sizeof(float));
    
   
    /* Testing
    a_nrows = 3; a_ncols = 2;
    b_nrows = 2; b_ncols = 4;
    csize = a_nrows*b_ncols;
    
    float* A = (float*) malloc(asize*sizeof(float));
    float* B = (float*) malloc(bsize*sizeof(float));
    float* C = (float*) malloc(csize*sizeof(float));
       
    A[0*a_ncols + 0] = 7;
    A[0*a_ncols + 1] = 8;
    A[1*a_ncols + 0] = 4;  
    A[1*a_ncols + 1] = 5;
    A[2*a_ncols + 0] = 7;
    A[2*a_ncols + 1] = 10;
    
    B[0*b_ncols + 0] = 8;
    B[0*b_ncols + 1] = 1;
    B[0*b_ncols + 2] = 0;
    B[0*b_ncols + 3] = 1;
    B[1*b_ncols + 0] = 4;
    B[1*b_ncols + 1] = 7;
    B[1*b_ncols + 2] = 2;
    B[1*b_ncols + 3] = 1;
    * 
    * */
    

        
    for(int i = 0; i < asize; ++i)
        A[i] = 1;    
        
    for(int i = 0; i < bsize; ++i)
        B[i] = 2;
       
    for(int i = 0; i < csize; ++i)
        C[i] = 0;
        
    //printMat(A, a_nrows, a_ncols);
    //printMat(B, b_nrows, b_ncols);    
        
    matrixMul(A, B, C, a_nrows, a_ncols, b_nrows, b_ncols);
    //printMat(C, a_nrows, b_ncols);
    
    
    float* D = (float*) malloc(asize*sizeof(float));
    
   /* float mask[] = { 1, 1, 1, 1, 1,
                     1, 2, 2, 2, 1,
                     1, 2, 3, 2, 1,
                     1, 2, 2, 2, 2,
                     1, 1, 1, 1, 1};
                     * */
                     
    float mask[] = { 1, 1, 1,
                     1, 3, 1,
                     1, 1, 1};

  //float mask[] = {2};
  
    int masksize = 3;
  
    printMat(mask, masksize, masksize);
    
    convolution2D(A, D, mask,  a_nrows, a_ncols, masksize);
    
    //printMat(A, a_nrows, a_ncols);
    //printMat(D, a_nrows, a_ncols);
    
    
    freeMemory( (void**) &A);
    freeMemory( (void**) &B);
    freeMemory( (void**) &C);
    freeMemory( (void**) &D);
    
 
#endif
    
    
}


