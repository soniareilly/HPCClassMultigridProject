#include <stdio.h>

inline double r(double h, double k) {
    return 0.5*k/(h*h);
}

inline double a(double v, double nu, double h, double r) {
    return r*(-v*h/2.0+nu);
}

inline double b(double v, double nu, double h, double r) {
    return r*(v*h/2.0+nu);
}

#define R(dt,dx)             (0.5*(dt)/((dx)*(dx)))
#define COEFF(v, nu, dx, r)  ((r)*(0.5*(v)*(dt)+(nu)))
#define CEIL(x,y)            (((x) + (y) - 1)/(y))

// Red-Black Gauss-Seidel
__global__ void gs_ker(double* u, double* rhs, 
                       long N, 
                       double* v1, double* v2, 
                       double dt, double nu, double dx,
                       int rb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i+j)%2 == rb){
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
        u[i*n+j] = (rhs[i*n+j] + c*up + d*dn + a*lf + b*rt)/(1.0-4.0*r*nu);
        }
    }
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

void gauss_seidel2(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h) 
{
    int i; int j;

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    // UPDATING RED POINTS
    // here we're assuming top left is red
    // interior
    for (i = 1; i < n; i++) {
    	for (j = 2-i%2; j < n; j+=2) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - dd*u[(i+1)*(n+1)+j] - aa*u[i*(n+1)+j-1] - bb*u[i*(n+1)+j+1])/(1.0-4.0*rr*nu);
		}
    }

    // UPDATING BLACK POINTS
    // interior
    for (i = 1; i < n; i++) {
    	for (j = 1+i%2; j < n; j+=2) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - dd*u[(i+1)*(n+1)+j] - aa*u[i*(n+1)+j-1] - bb*u[i*(n+1)+j+1])/(1.0-4.0*rr*nu);
		}
    }
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
    rhs = (double*) calloc (sizeof(double) * n*n);

    double *cuu, *cuv1, *cuv2, *curhs;
    cudaMalloc(&cuu  , sizeof(double) * n*n);
    cudaMalloc(&cuv1 , sizeof(double) * n*n);
    cudaMalloc(&cuv2 , sizeof(double) * n*n);
    cudaMalloc(&curhs, sizeof(double) * n*n);
    cudaMemset(curhs, 0, sizeof(double) * n*n);

    // initialize u0, v1, v2
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            // Gaussian initial condition u0
            u0[i*(N+1)+j] = exp(-sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
            
            // rotating velocity field
            v1[i*(N+1)+j] = -ky*sin(kx*i*dx)*cos(ky*j*dx);
            v2[i*(N+1)+j] = kx*cos(kx*i*dx)*sin(ky*j*dx);
        }
    }
    cudaMemcpy(cuu , u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv1, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuv2, u, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    gauss_seidel2(u, rhs, N, v1, v2, dt, nu, dx);
    dim3 numthreads(N,N);   // DO NOT MAKE THIS LARGER THAN 32
    gauss_seidel<<<numthreads,1>>>(cuu, curhs, N, cuv1, cuv2, dt, nu, dx);
    cudaMemcpy(rhs, cuu, sizeof(double)*n*n, cudaMemcpyDeviceToHost);

    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%g\t",u[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            printf("%g\t",rhs[i*n+j]);
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