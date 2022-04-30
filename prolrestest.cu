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

__global__ void square_ker(double* a, long n)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)    a[i] *= a[i];
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

double compute_norm(double *res, long n) 
{
    double tmp = 0.0;
    for (long i = 1; i < n; i++)
    {
        for (long j = 1; j < n; j++) 
        {
            tmp += res[i*(n+1)+j] * res[i*(n+1)+j]; 
        }
    }
    return sqrt(tmp);
}

int main()
{
    int N = 8;
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

            //rhs[i*(N+1)+j] = 0.0;
        }
    }
    cudaMemcpy(cuu , u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv1, v1, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv2, v2, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    //cudaMemcpy(curhs, rhs, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    compute_rhs(rhs, u, N, v1, v2, dt, nu, dx);
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(CEIL(n,threadsPerBlock.x), CEIL(n,threadsPerBlock.y));
    compute_rhs_cu<<<numBlocks,threadsPerBlock>>>(curhs, cuu, N, cuv1, cuv2, dt, nu, dx);
    cudaMemcpy(v1, curhs, sizeof(double)*n*n, cudaMemcpyDeviceToHost);

    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%.4g  ",rhs[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%.4g  ",v1[i*n+j]);
        }
        printf("\n");
    }

    free(u);
    free(v1);
    free(v2);
    free(rhs);
    cudaFree(cuu);
    cudaFree(cuv1);
    cudaFree(cuv2);
    cudaFree(curhs);
}