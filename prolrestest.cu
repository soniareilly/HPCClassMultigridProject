#include <stdio.h>
//#include "gscu.h"
//#include "gs.h"

__global__ void restriction(double *u, double *up, int n)
{
    // u is the output, size n/2+1 x n/2+1, up is the input, size n+1 x n+1
    // Call this with a grid of n/2+1 x n/2+1 threads
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int nnew = n/2 + 1;
    if ((ix < nnew) && (iy < nnew))
        u[ix*nnew+iy] = up[(2*ix)*(n+1)+(2*iy)];
}

// Restriction test
int main()
{
    int N = 8;
    int n = N+1;
    int n2 = N/2+1;
    int i,j;
    double *arr = (double*) malloc ( sizeof(double) * n * n );
    double *sol = (double*) malloc ( sizeof(double) * n2 * n2 );
    for (i = 0; i < n*n; ++i) arr[i] = i;
    double *cuarr, *cusol;
    cudaMalloc(&cuarr, sizeof(double) * n*n);
    cudaMalloc(&cusol, sizeof(double) * n2*n2);
    cudaMemcpy(cuarr, arr, n*n*sizeof(double), cudaMemcpyHostToDevice);

    dim3 numthreads(n2,n2);
    restriction<<<numthreads,1>>>(cusol,cuarr,N);
    cudaMemcpy(sol, cusol, n2*n2*sizeof(double), cudaMemcpyDeviceToHost);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
             printf("%g\t",arr[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < n2; ++i)
    {
        for (j = 0; j < n2; ++j)
        {
            printf("%g\t",sol[i*n2+j]);
        }
        printf("\n");
    }

    free(arr);
    free(sol);
    cudaFree(cuarr);
    cudaFree(cusol);
}