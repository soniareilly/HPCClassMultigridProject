#include <stdio.h>

#define R((dt),(dx))              (0.5*(dt)/((dx)*(dx)))
#define COEFF((v),(nu),(dx),(r))) ((r)*(0.5*(v)*(dt)+(nu)))
#define CEIL(x,y)                 (((x) + (y) - 1)/(y))

__global__ void square_ker(double* a, long n)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)    a[i] *= a[i];
}

__global__ void gpucopy(double* dest, const double* source, long n)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)    dest[i] = source[i];
}

__global__ void reduction_kernel(double* sum, const double* a, long N)
{
    // Courtesy of B. Peherstorfer
    __shared__ double smem[1024];
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    // each thread reads data from global into shared memory
    if (idx < N) smem[threadIdx.x] = a[idx];
    else smem[threadIdx.x] = 0;

    __syncthreads();

    for (unsigned int s = blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    // write to global memory
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

double compute_norm(double* a, int N)
{
    long n = (N+1)*(N+1);
    long nb = CEIL(n,1024); 
    square_ker<<<nb,1024>>>(a,n);
    long nt = n; // num threads
    do
    {
        nb = CEIL(nt,1024); // num blocks
        reduction_kernel<<<nb,1024>>>(a,a,nt);
        nt = nb;  // num threads for next iteration is num blocks for this one
    } while (nb > 1);
    double norm;
    cudaMemcpy(&norm, a, 1*sizeof(double), cudaMemcpyDeviceToHost);
}


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

__global__ void residual(double* r, double* u, double* rhs, 
                         int N, 
                         double* v1, double* v2, 
                         double dt, double nu, double dx)
{
    int n = N+1;

    __shared__ double uloc[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n) && (j < n)) {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = u[i*n+j];
    } else {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = 0.0;
    }
}

// Kernel to perform Jacobi iteration
__global__ void jacobi_kernel(double* u, double* rhs, 
                              int N, 
                              double* v1, double* v2, 
                              double dt, double nu, double dx)
{
    // NOTE!! n = N+1
    int n = N+1;
    // copy a chunk into shared memory
    __shared__ double uloc[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n) && (j < n)) {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = u[i*n+j];
    } else {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = 0.0;
    }

    __syncthreads();

    double v1i = v1[i*n+j];
    double v2i = v2[i*n+j];
    double r = R(dt, dx);
    double a = COEFF(-v2i,nu,dx,r);
    double b = COEFF( v2i,nu,dx,r);
    double c = COEFF(-v1i,nu,dx,r);
    double d = COEFF( v1i,nu,dx,r);

    double up = 0.0;
    double down = 0.0;
    double left = 0.0;
    double right = 0.0;

    if (threadIdx.x > 0) {
    up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.y > 0) {
    left = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
    } else if (j > 0) {
    left = u[i*n+j-1];
    }

    if (threadIdx.x < blockDim.x-1) {
    down = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
    } else if (i < n) {
    down = u[(i+1)*n+j];
    }

    if (threadIdx.y < blockDim.y-1) {
    right = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
    } else if (j < n) {
    right = u[i*n+j+1];
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
    u[i*n+j] = 0.25*(dx*dx*f[i*n+j] + c*up + d*down + a*left + b*right);
    }
}

// Red-Black Gauss-Seidel
__global__ void gauss_seidel(double* u, double* rhs, 
                            long N, 
                            double* v1, double* v2, 
                            double dt, double nu, double dx)
{
    // Call with at least (N/2+1) x (N/2+1) threads
    // NOTE!! n = N+1
    long n = N+1;

    // copy a chunk into shared memory
    __shared__ double uloc[1024];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n) && (j < n)) {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = u[i*n+j];
    } else {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = 0.0;
    }

    __syncthreads();

    double v1i = v1[i*n+j];
    double v2i = v2[i*n+j];
    double r = R(dt, dx);
    double a = COEFF(-v2i,nu,dx,r);
    double b = COEFF( v2i,nu,dx,r);
    double c = COEFF(-v1i,nu,dx,r);
    double d = COEFF( v1i,nu,dx,r);

    double up = 0.0;
    double down = 0.0;
    double left = 0.0;
    double right = 0.0;

    if (threadIdx.x > 0) {
    up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.y > 0) {
    left = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
    } else if (j > 0) {
    left = u[i*n+j-1];
    }

    if (threadIdx.x < blockDim.x-1) {
    down = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
    } else if (i < n) {
    down = u[(i+1)*n+j];
    }

    if (threadIdx.y < blockDim.y-1) {
    right = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
    } else if (j < n) {
    right = u[i*n+j+1];
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
    u[i*n+j] = 0.25*(dx*dx*f[i*n+j] + c*up + d*down + a*left + b*right);
    }
}