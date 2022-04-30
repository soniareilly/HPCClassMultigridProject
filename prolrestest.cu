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

__global__ void residual_cu(double* res, double* u, double* rhs, 
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
    res[i*n+j] = (rhs[i*n+j] - (1.0-4.0*r*nu)*uij + c*up + d*dn + a*lf + b*rt);
}

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);
    // printf("num threads:%d\n",omp_get_num_threads());

    // #pragma omp parallel num_threads(ntr)
{
    // #pragma omp for collapse(2)
    for (long i = 1; i < n; i++){
        //#pragma omp task
{
        for (long j = 1; j < n; j++) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            res[i*(n+1)+j] = rhs[i*(n+1)+j] - ((1.0-4.0*rr*nu)*u[i*(n+1)+j] + cc*u[(i-1)*(n+1)+j] + aa*u[i*(n+1)+(j-1)] + dd*u[(i+1)*(n+1)+j] + bb*u[i*(n+1)+(j+1)]);
            
        }
}
    }
    //#pragma omp taskwait
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

    double *u, *v1, *v2, *rhs, *res;
    u   = (double*) malloc (sizeof(double) * n*n);
    v1  = (double*) malloc (sizeof(double) * n*n);
    v2  = (double*) malloc (sizeof(double) * n*n);
    rhs = (double*) malloc (sizeof(double) * n*n);
    res = (double*) malloc (sizeof(double) * n*n);

    double *cuu, *cuv1, *cuv2, *curhs, *cures;
    cudaMalloc(&cuu  , sizeof(double) * n*n);
    cudaMalloc(&cuv1 , sizeof(double) * n*n);
    cudaMalloc(&cuv2 , sizeof(double) * n*n);
    cudaMalloc(&curhs, sizeof(double) * n*n);
    cudaMalloc(&cures, sizeof(double) * n*n);

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
    cudaMemcpy(cuu , u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv1, v1, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv2, v2, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(curhs, rhs, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    residual(res, u, rhs, N, v1, v2, dt, nu, dx);
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(CEIL(n,threadsPerBlock.x), CEIL(n,threadsPerBlock.y));
    residual_cu<<<numBlocks,threadsPerBlock>>>(cures, cuu, curhs, N, cuv1, cuv2, dt, nu, dx);
    cudaMemcpy(rhs, cures, sizeof(double)*n*n, cudaMemcpyDeviceToHost);

    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%.4g  ",res[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%.4g  ",rhs[i*n+j]);
        }
        printf("\n");
    }

    free(u);
    free(v1);
    free(v2);
    free(rhs);
    free(res);
    cudaFree(cuu);
    cudaFree(cuv1);
    cudaFree(cuv2);
    cudaFree(curhs);
    cudaFree(cures);
}