#include <stdio.h>

#include "utils.h"

void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	printf("%s returned %s at %s: %d\n", statement, cudaGetErrorString(err), file,line);
	exit (1);
}


void freeMemory(void** ptr)
{
    if(*ptr != 0)
    {
        free(*ptr);
        *ptr = 0;
    }
}
