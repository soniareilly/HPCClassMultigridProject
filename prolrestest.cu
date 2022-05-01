#include <stdio.h>

#define PI 3.1415926535897932
#define R(dt,dx)             (0.5*(dt)/((dx)*(dx)))
#define COEFF(v, nu, dx, r)  ((r)*(0.5*(v)*(dx)+(nu)))
#define CEIL(x,y)            (((x) + (y) - 1)/(y))

inline double r(double h, double k) {
    return 0.5*k/(h*h);
}

inline double a(double v, double nu, double h, double r) {
    return r*(-v*h/2.0+nu);
}

inline double b(double v, double nu, double h, double r) {
    return r*(v*h/2.0+nu);
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

// Kernel to initialize u0 as a Gaussian
__global__ void gaussian_u0(double* u0, double x0, double y0, double sigma, int n, double dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i > 0) && (j > 0) && (i < n) && (j < n)){
        u0[i*(n+1)+j] = exp(-sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
    } else if ((i == 0) || (j == 0) || (i == n-1) || (j == n-1)){
        u0[i*(n+1)+j] = 0.0;
    }
}

int main()
{
    int N = 2048;
    int n = N+1;
    double dx = 1.0/n;
    double dt = dx/10.0;
    double nu = -4*1e-4;
    int i,j;

    double x0 = 0.2, y0 = 0.4;
    double sigma = 100.0;
    double kx = 1.0*PI;
    double ky = 1.0*PI;

    double *u, *v1, *v2, *rhs;
    u   = (double*) malloc (sizeof(double) * n*n);
    v1  = (double*) malloc (sizeof(double) * n*n);
    v2  = (double*) malloc (sizeof(double) * n*n);
    rhs = (double*) malloc (sizeof(double) * n*n);

    double *cuu, *cuv1, *cuv2, *curhs;
    cudaMalloc(&cuu  , sizeof(double) * n*n);
    cudaMalloc(&cuv1 , sizeof(double) * n*n);
    cudaMalloc(&cuv2 , sizeof(double) * n*n);
    cudaMalloc(&curhs, sizeof(double) * n*n);

    // initialize u0, v1, v2
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            // Gaussian initial condition u0
            u[i*(N+1)+j] = exp(-sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
            
            // rotating velocity field
            v1[i*(N+1)+j] = -ky*sin(kx*i*dx)*cos(ky*j*dx);
            v2[i*(N+1)+j] = kx*cos(kx*i*dx)*sin(ky*j*dx);

            rhs[i*(N+1)+j] = 0.0;
        }
    }
    for (i = 0; i < N; ++i)
    {
        u[i] = 0.0;
        u[i*n+N] = 0.0;
        u[N*n+i+1] = 0.0;
        u[i*n] = 0.0;
    }
    cudaMemcpy(cuu , u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv1, v1, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv2, v2, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(curhs, rhs, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(CEIL(n,threadsPerBlock.x), CEIL(n,threadsPerBlock.y));



    float ms;
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    int ntrials = 500;
    cudaEventRecord(startEvent,0);
    for (i = 0; i < ntrials; ++i)
    {
        gauss_seidel(cuu, curhs, n, cuv1, cuv2, dt, nu, dx);
    }
    cudaEventRecord(stopEvent,0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    int ops = 31*N*N*ntrials;
    printf("Flops: %f\n", 1000*ops/ms);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    free(u);
    free(v1);
    free(v2);
    free(rhs);
    cudaFree(cuu);
    cudaFree(cuv1);
    cudaFree(cuv2);
    cudaFree(curhs);
}