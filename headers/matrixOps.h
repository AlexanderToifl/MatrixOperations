//matrixOps.h
#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

//only for small matrices
void printMat(float *A, int dim_x, int dim_y);

void matrixMul(float* A, float* B, float* C, int Adim_x, int Adim_y, int Bdim_x, int Bdim_y);

void squareMatrixAdd(float* A, float* B, float* C, int dim);
void squareMatrixMul(float* A, float* B, float* C, int dim);

//convolution with square mask
void convolution2D(float* in, float* out, float* mask, int in_nrows, int in_ncols, int mask_dim);


#endif
