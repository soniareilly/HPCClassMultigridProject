#include <stdio.h>

#define R(dt,dx)             (0.5*(dt)/((dx)*(dx)))
#define COEFF(v, nu, dx, r)  ((r)*(0.5*(v)*(dx)+(nu)))
#define CEIL(x,y)            (((x) + (y) - 1)/(y))

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
    return norm;
}


__global__ void prolongation(double* up, double* u, int n)
{
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    // Call with grid of n+1 x n+1 threads
    int ix = blockIdx.x*blockDim.x + threadIdx.x; int ix2 = 2*ix;
    int iy = blockIdx.y*blockDim.y + threadIdx.y; int iy2 = 2*iy;
    int nnew = 2*n + 1;
    if ((ix2 >= nnew) || (iy2 >= nnew)) return; // Out of bounds
    up[ix2*nnew+iy2] = u[ix*(n+1)+iy];
    if (ix2 < nnew-1){
        up[(ix2+1)*nnew + iy2] = 0.5*( u[ix*(n+1) + iy] + u[(ix+1)*(n+1) + iy] );
        if (iy2 < nnew-1){
            up[(ix2+1)*nnew+iy2+1] = 0.25*(u[ix*(n+1)+iy]+u[(ix+1)*(n+1)+iy]+u[ix*(n+1)+iy+1]+u[(ix+1)*(n+1)+iy+1]);
        }
    }
    if (iy2 < nnew-1){
        up[ix2*nnew+iy2+1] = 0.5*(u[ix*(n+1)+iy]+u[ix*(n+1)+iy+1]);
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
        u[ix*nnew+iy] = up[(2*ix)*(n+1)+(2*iy)];
}

__global__ void compute_rhs(double* rhs, double* u, 
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

    __syncthreads();

    double v1i = v1[i*n+j];
    double v2i = v2[i*n+j];
    double r = R(dt, dx);
    double a = COEFF(-v2i,nu,dx,r);
    double b = COEFF( v2i,nu,dx,r);
    double c = COEFF(-v1i,nu,dx,r);
    double d = COEFF( v1i,nu,dx,r);

    double up = 0.0;
    double dn = 0.0;
    double lf = 0.0;
    double rt = 0.0;

    if (threadIdx.x > 0) {
    up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.y > 0) {
    lf = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
    } else if (j > 0) {
    lf = u[i*n+j-1];
    }

    if (threadIdx.x < blockDim.x-1) {
    dn = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
    } else if (i < n-1) {
    dn = u[(i+1)*n+j];
    }

    if (threadIdx.y < blockDim.y-1) {
    rt = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
    } else if (j < n-1) {
    rt = u[i*n+j+1];
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
        double uij = uloc[threadIdx.x*blockDim.y + threadIdx.y];
    rhs[i*n+j] = (1.0+4.0*r*nu)*uij - c*up - d*dn - a*lf - b*rt;
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

    __syncthreads();

    double v1i = v1[i*n+j];
    double v2i = v2[i*n+j];
    double r = R(dt, dx);
    double a = COEFF(-v2i,nu,dx,r);
    double b = COEFF( v2i,nu,dx,r);
    double c = COEFF(-v1i,nu,dx,r);
    double d = COEFF( v1i,nu,dx,r);

    double up = 0.0;
    double dn = 0.0;
    double lf = 0.0;
    double rt = 0.0;

    if (threadIdx.x > 0) {
    up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.y > 0) {
    lf = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
    } else if (j > 0) {
    lf = u[i*n+j-1];
    }

    if (threadIdx.x < blockDim.x-1) {
    dn = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
    } else if (i < n-1) {
    dn = u[(i+1)*n+j];
    }

    if (threadIdx.y < blockDim.y-1) {
    rt = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
    } else if (j < n-1) {
    rt = u[i*n+j+1];
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
        double uij = uloc[threadIdx.x*blockDim.y + threadIdx.y];
    r[i*n+j] = (rhs[i*n+j] - (1.0-4.0*r*nu)*uij + c*up + d*dn + a*lf + b*rt);
}

// Kernel to initialize u0 as a Gaussian
__global__ void gaussian_u0(double* u0, double x0, double y0, double sigma, int n, double dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n+1) && (j < n+1)){
        u0[i*(n+1)+j] = exp(-sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
    }
}

// Kernel to initialize v1 and v2 as a rotating field
__global__ void rotating_v(double* u0, double kx, double ky, int n, double dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < n+1) && (j < n+1)){
        v1[i*(N+1)+j] = -ky*sin(kx*i*dx)*cos(ky*j*dx);
        v2[i*(N+1)+j] = kx*cos(kx*i*dx)*sin(ky*j*dx);
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
    double dn = 0.0;
    double lf = 0.0;
    double rt = 0.0;

    if (threadIdx.x > 0) {
    up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
    } else if (i > 0) {
    up = u[(i-1)*n+j];
    }

    if (threadIdx.y > 0) {
    lf = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
    } else if (j > 0) {
    lf = u[i*n+j-1];
    }

    if (threadIdx.x < blockDim.x-1) {
    dn = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
    } else if (i < n-1) {
    dn = u[(i+1)*n+j];
    }

    if (threadIdx.y < blockDim.y-1) {
    rt = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
    } else if (j < n-1) {
    rt = u[i*n+j+1];
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
    u[i*n+j] = (rhs[i*n+j] - c*up - d*dn - a*lf - b*rt)/(1.0-4.0*r*nu);
    }
}

__global__ void gs_ker(double* u, double* rhs, 
                       long N, 
                       double* v1, double* v2, 
                       double dt, double nu, double dx,
                       int rb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // NOTE!! n = N+1
    long n = N+1;

    double v1i = v1[i*n+j];
    double v2i = v2[i*n+j];
    double r = R(dt, dx);
    double a = COEFF(-v2i,nu,dx,r);
    double b = COEFF( v2i,nu,dx,r);
    double c = COEFF(-v1i,nu,dx,r);
    double d = COEFF( v1i,nu,dx,r);

    double up = 0.0;
    double dn = 0.0;
    double lf = 0.0;
    double rt = 0.0;

    // copy a chunk into shared memory
    __shared__ double uloc[1024];
    if ((i < n) && (j < n)) {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = u[i*n+j];
    } else {
    uloc[threadIdx.x*blockDim.y + threadIdx.y] = 0.0;
    }

    __syncthreads();

    double unew = uloc[threadIdx.x*blockDim.y + threadIdx.y];
    if ((i+j)%2 == rb){
        if (threadIdx.x > 0) {
        up = uloc[(threadIdx.x-1)*blockDim.y + threadIdx.y];
        } else if (i > 0) {
        up = u[(i-1)*n+j];
        }

        if (threadIdx.y > 0) {
        lf = uloc[threadIdx.x*blockDim.y + threadIdx.y-1];
        } else if (j > 0) {
        lf = u[i*n+j-1];
        }

        if (threadIdx.x < blockDim.x-1) {
        dn = uloc[(threadIdx.x+1)*blockDim.y + threadIdx.y];
        } else if (i < n-1) {
        dn = u[(i+1)*n+j];
        }

        if (threadIdx.y < blockDim.y-1) {
        rt = uloc[threadIdx.x*blockDim.y + threadIdx.y+1];
        } else if (j < n-1) {
        rt = u[i*n+j+1];
        }

        if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1)) {
        unew = (rhs[i*n+j] - c*up - d*dn - a*lf - b*rt)/(1.0-4.0*r*nu);
        }
    }

    __syncthreads();

    if ((i > 0) && (i < n-1) && (j > 0) && (j < n-1))    u[i*n+j] = unew;
}

// Red-Black Gauss-Seidel
void gauss_seidel(double* u, double* rhs, 
                  long N, 
                  double* v1, double* v2, 
                  double dt, double nu, double dx)
{
    // calling kernel with N instead of N+1, since bottom & right borders are all 0's
    // allows u to fit in fewer blocks, since N is a power of 2
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(CEIL(N,threadsPerBlock.x), CEIL(N,threadsPerBlock.y));
    // red points
    gs_ker<<<numBlocks, threadsPerBlock>>>(u,rhs,N,v1,v2,dt,nu,dx,0);
    // black points
    gs_ker<<<numBlocks, threadsPerBlock>>>(u,rhs,N,v1,v2,dt,nu,dx,1);
}