#include <stdio.h>

__global__ void prolongation(double* up, double* u, int n)
{
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    // Call with grid of n+1 x n+1 threads
    int ix = blockIdx.x*blockDim.x + threadIdx.x; int ix2 = 2*ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; int iy2 = 2*iy;
    int nnew = 2*n + 1;
    if ((ix2 <= nnew) || (iy2 <= nnew)) return; // Out of bounds
    up[ix2*nnew+iy2] = u[ix*(n+1)+iy];
    if (ix2 < nnew-1){
        up[(ix2+1)*nnew+iy2] = 0.5*(u[ix*(n+1)+iy]+u[(ix+1)*(n+1)+iy]);
        if (iy2 < nnew-1)){
            up[(ix2+1)*nnew+iy2+1] = 0.25*(u[ix*(n+1)+iy]+u[(ix+1)*(n+1)+iy]+u[ix*(n+1)+iy+1]+u[(ix+1)*(n+1)+iy+1]);
        }
    }
    if (iy2 < nnew-1){
        up[ix2*nnew+iy2+1] = 0.5*(u[ix*(n+1)+iy]+u[ix(n+1)+iy+1]);
    }
}

__global__ void restriction(double *u, double *up, int n)
{
    // u is the output, size n/2+1 x n/2+1, up is the input, size n+1 x n+1
    // Call this with a grid of n/2+1 x n/2+1 threads
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int nnew = n/2 + 1;
    if ((ix < nnew) && (iy < nnew))
        u[ix*nnew+iy] = up[(2*ix-1)*(n+1)+(2*iy-1)];
}

// Kernel to perform Jacobi iteration
__global__ void jacobi_kernel(long N, double *f, double *u){
    // NOTE!! n = N+1
    long n = N+1;
    // copy a chunk into shared memory
    __shared__ double uloc[1024];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n) && (j < n)) {
    uloc[threadIdx.y*blockDim.x + threadIdx.x] = u[i*n+j];
    } else {
    uloc[threadIdx.y*blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    double up = 0.0;
    double down = 0.0;
    double left = 0.0;
    double right = 0.0;

    if (threadIdx.y > 0) {
    up = uloc[(threadIdx.y-1)*blockDim.x + threadIdx.x];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.x > 0) {
    left = uloc[threadIdx.y*blockDim.x + threadIdx.x-1];
    } else if (j > 0) {
    left = u[i*n+j-1];
    }

    if (threadIdx.y < blockDim.y-1) {
    down = uloc[(threadIdx.y+1)*blockDim.x + threadIdx.x];
    } else if (i < n) {
    down = u[(i+1)*n+j];
    }

    if (threadIdx.x < blockDim.x-1) {
    right = uloc[threadIdx.y*blockDim.x + threadIdx.x+1];
    } else if (j < n) {
    right = u[i*n+j+1];
    }

    __syncthreads();

    if ((i < n) && (j < n)) {
    u[i*n+j] = (f[i*n+j]/n/n + up + down + left + right)/4;
    }
}