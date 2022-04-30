#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#define PI 3.1415926535897932

// u is row major

// functions for computing coefficients for LHS (and RHS)
inline double r(double h, double k) {
    return 0.5*k/(h*h);
}


inline double a(double v, double nu, double h, double r) {
    return r*(-v*h/2.0+nu);
}

inline double b(double v, double nu, double h, double r) {
    return r*(v*h/2.0+nu);
}

void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    //#pragma omp parallel num_threads(ntr)
{
    //#pragma omp for collapse(2)
    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            rhs[i*(n+1)+j] = (1.0+4.0*rr*nu)*u[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)];
            //printf("rhs:%.6f\n", rhs[i*(n+1)+j]);
            
        }
    }
}
    //#pragma omp barrier

}

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);
    // printf("num threads:%d\n",omp_get_num_threads());

    //#pragma omp parallel num_threads(ntr)
{
    //#pragma omp for collapse(2) ordered
    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            res[i*(n+1)+j] = rhs[i*(n+1)+j] - ((1.0-4.0*rr*nu)*u[i*(n+1)+j] + cc*u[(i-1)*(n+1)+j] + aa*u[i*(n+1)+(j-1)] + dd*u[(i+1)*(n+1)+j] + bb*u[i*(n+1)+(j+1)]);
            
        }
    }
}
   //#pragma omp barrier

}


double compute_norm(double *res, long n, int ntr) {

    double tmp = 0.0;

    //#pragma omp parallel num_threads(ntr)
{
    //#pragma omp for collapse(2) //ordered //reduction(+: tmp)
    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            
            //#pragma omp atomic update
            tmp += res[i*(n+1)+j] * res[i*(n+1)+j];
            
        }
    }
}
    return sqrt(tmp);
}

void gauss_seidel(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    #pragma omp parallel num_threads(ntr)
{ 
        // GS iteration to update red points
        #pragma omp for collapse(2) nowait
        for (long i=1; i < n; i+=2) { // iterate through every odd row of red points (starting at 1)
                for (long j=1; j < n; j+=2) { // iterate through every other column of red points at each row
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    // printf("a:%.6f\n",aa);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
                
        }

        #pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of red points (starting at 2)
                for (long j=2; j < n; j+=2) {
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
                
        }
	
        // GS iteration to update black points
        #pragma omp for collapse(2) nowait
        for (long i=1; i < n; i+=2) { // iterate through every odd row of black points (starting at 2)
            for (long j=2; j < n; j+=2) { // iterate through every other column of black points at each row
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
        }

        #pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of black points (starting at 1)
                for (long j=1; j < n; j+=2) {
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
        }
}
        // double* utmp = u;
        // u = unew;
        // unew = utmp;


}

void gauss_seidel2(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {

    int i; int j;

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    // UPDATING RED POINTS
    // here we're assuming top left is red
    // interior
    #pragma omp parallel for private(i,j) num_threads(ntr)
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
    #pragma omp parallel for private(i,j) num_threads(ntr)
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

int main(){
    int N = 5;
    int ntr = 16;
    
    double *u, *res, *rhs, *v1, *v2;
    res = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    rhs = (double*) calloc ( sizeof(double), (N+1)*(N+1) );
    u = (double*) malloc ( sizeof(double)* (N+1)*(N+1) );
    v1 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    v2 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );


    double nu = -4*1e-4;
    double dx = 1.0/N;
    double dt = dx/10;
    double x0 = 0.2, y0 = 0.4;
    double sigma = 100.0;
    double kx = 1.0*PI;
    double ky = 1.0*PI;
    // Initialize u
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N+1; i++){
        for(int j = 0; j < N+1; j++){

            // Gaussian initial condition u0
            u[i*(N+1)+j] = exp(-sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
            
            // rotating velocity field
            v1[i*(N+1)+j] = -ky*sin(kx*i*dx)*cos(ky*j*dx);
            v2[i*(N+1)+j] = kx*cos(kx*i*dx)*sin(ky*j*dx);

        }
    }
    //printf("yes\n");
    double tt = omp_get_wtime();
    compute_rhs(rhs, u, N, v1, v2, dt, nu, dx, ntr);
    //printf("yes\n");
    //#pragma omp barrier
    residual(res, u, rhs, N, v1, v2, dt, nu, dx, ntr);
    double res_norm = compute_norm(res, N, ntr);
    gauss_seidel2(u, rhs, N, v1, v2, dt, nu, dx, ntr);
    printf("\nCPU (%d threads) time: %f s\n", ntr,omp_get_wtime()-tt);

    printf("\nu matrix\n");
    for (int i = 0; i < N+1; ++i)
    {
        for (int j = 0; j < N+1; ++j)
        {
            printf("%.4f\t",u[i*(N+1)+j]);
        }
        printf("\n");
    }

    printf("\nrhs matrix\n");
    for (int i = 0; i < N+1; ++i)
    {   
        for (int j = 0; j < N+1; ++j)
        {
            printf("%.4f\t",rhs[i*(N+1)+j]);
        }
        printf("\n");
    }
    
    printf("\nres matrix\n");
    for (int i = 0; i < N+1; ++i)
    {   
        for (int j = 0; j < N+1; ++j)
        {
            printf("%.4f\t",res[i*(N+1)+j]);
        }
        printf("\n");
    }
    
    printf("res norm: %6f\n", res_norm);


    free(u);
    free(res);
    free(rhs);
    free(v1);
    free(v2);
}
