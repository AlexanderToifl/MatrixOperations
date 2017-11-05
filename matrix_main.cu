//matrix_main.cu


#include <stdio.h>
#include <stdlib.h>
#include <time.h>  


#include "utils.h"
#include "vectorOps.h"
#include "matrixOps.h"



int main( int argc, char **argv )
{
    if(argc < 2)
    {
        printf("1 argument required: integer number (vector size)\n");
        return 0;
    }
    
    int seed = time(NULL);
    srand(seed);
    
    int n = atoi(argv[1]);
    printf("n = %d\n", n);
    
    float* A = (float*) malloc(n*sizeof(float));
    float* B = (float*) malloc(n*sizeof(float));
    float* C = (float*) malloc(n*sizeof(float));
    
    int i=0;
    
    for(; i < n; ++i)
    {
        A[i] = rand() % 10 ;
        B[i] = rand() % 10 ;
        
    }
    
    vecAdd(A, B, C, n); 
    
    printf("%f\n",C[1]);
    
    freeMemory( (void**) &A);
    freeMemory( (void**) &B);
    freeMemory( (void**) &C);
    
    
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
    
}


