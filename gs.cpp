#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gs.h"
#include <omp.h>

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


// compute rhs with u
void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    // #pragma omp parallel num_threads(ntr)
{
    // #pragma omp for collapse(2)
    for (long i = 1; i < n; i++){
        #pragma omp task
{
        for (long j = 1; j < n; j++) {
            
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            rhs[i*(n+1)+j] = (1.0+4.0*rr*nu)*u[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)];
            // printf("rhs:%.6f\n", rhs[i*(n+1)+j]);
            
        }
}
    }
    #pragma omp taskwait
}

}

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
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
        #pragma omp task
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
    #pragma omp taskwait
}

}


double compute_norm(double *res, long n, int ntr) {

    double tmp = 0.0;

    // #pragma omp parallel num_threads(ntr)
{
    // #pragma omp for collapse(2) //ordered //reduction(+: tmp)
    for (long i = 1; i < n; i++){
        //#pragma omp task
{
        for (long j = 1; j < n; j++) {
            
            //#pragma omp atomic update
            tmp += res[i*(n+1)+j] * res[i*(n+1)+j];
            
        }
}
    }
    //#pragma omp taskwait
}
    return sqrt(tmp);
}

void gauss_seidel(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd,rr; // LHS coeffficients
    rr = r(h,k);

    //#pragma omp parallel num_threads(ntr)
{ 
        // GS iteration to update red points
        //#pragma omp for collapse(2) nowait
        for (long i=1; i < n; i+=2) { // iterate through every odd row of red points (starting at 1)
                #pragma omp task
{//printf("i:%d, thread id:%d\n",i,omp_get_thread_num());
                for (long j=1; j < n; j+=2) { // iterate through every other column of red points at each row
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    // printf("a:%.6f\n",aa);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
}
                
        }

        //#pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of red points (starting at 2)
                #pragma omp task
{
                for (long j=2; j < n; j+=2) {
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
}               
        }
        #pragma omp taskwait
	
        // GS iteration to update black points
        //#pragma omp for collapse(2) nowait
        for (long i=1; i < n; i+=2) { // iterate through every odd row of black points (starting at 2)
                #pragma omp task
{
                for (long j=2; j < n; j+=2) { // iterate through every other column of black points at each row
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);                   
                }
}
        }

        //#pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of black points (starting at 1)
                #pragma omp task
{
                for (long j=1; j < n; j+=2) {
                    
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    u[i*(n+1)+j] = (rhs[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)])/(1.0-4.0*rr*nu);
                    
                }
}
        }
}
        #pragma omp taskwait
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

void prolongation(double* up, double* u, int n, int ntr){
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    //#pragma omp parallel num_threads(ntr)
{
    //#pragma omp for collapse(2)
    for (int i = 0; i < n; i++){
        #pragma omp task
{
        for (int j = 0; j < n; j++){
            
            up[2*i*(2*n+1) + 2*j] = u[i*(n+1) + j];
            up[(2*i+1)*(2*n+1) + 2*j] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j])/2;
            up[2*i*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[i*(n+1) + j+1])/2;
            up[(2*i+1)*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j] + u[i*(n+1) + j+1] + u[(i+1)*(n+1) + j+1])/4;
            
        }
}
    }

    // right and bottom borders
    //#pragma omp for
    #pragma omp task
{
    for (int i = 0; i < n; i++){
        // right border
        
        up[2*i*(2*n+1) + 2*n] = u[i*(n+1) + n];
        up[(2*i+1)*(2*n+1) + 2*n] = (u[i*(n+1) + n] + u[(i+1)*(n+1) + n])/2;
        // bottom border
        up[2*n*(2*n+1) + 2*i] = u[n*(n+1) + i];
        up[2*n*(2*n+1) + 2*i+1] = (u[n*(n+1) + i] + u[n*(n+1) + i+1])/2;
        
    }
}
    #pragma omp taskwait
}
    // bottom right corner
    up[(2*n)*(2*n+1) + 2*n] = u[n*(n+1) + n];
}

void restriction(double* u, double* up, int n, int ntr){
    // u is the output, size n/2+1 x n/2+1, up is the input, size n+1 x n+1
    //#pragma omp parallel num_threads(ntr)
{
    //#pragma omp for collapse(2)
    for (int i = 0; i < n/2+1; i++){
        #pragma omp task
{
        for (int j = 0; j < n/2+1; j++){
            // More complicated way to do a restriction -- get this working if time permits
            //u[i*(n/2+1)+j] = (up[(2*i-1)*(n+1)+2*j-1] + 2*up[(2*i-1)*(n+1)+2*j] + up[(2*i-1)*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (2*up[2*i*(n+1)+2*j-1] + 4*up[2*i*(n+1)+2*j] + 2*up[2*i*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (up[(2*i+1)*(n+1)+2*j-1] + 2*up[(2*i+1)*(n+1)+2*j] + up[(2*i+1)*(n+1)+2*j+1])/16;
            // Simpler version 
            
            u[i*(n/2+1)+j] = up[2*i*(n+1)+2*j];
            // printf("u:%.6f\n",u[i*(n/2+1)+j]);
            
        }
}
    }
}
    #pragma omp taskwait

}
