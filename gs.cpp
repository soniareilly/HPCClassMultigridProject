#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gs.h"

// functions for computing coefficients for LHS (and RHS)
inline double r(double h, double k) {
    return 1.0/2*k/(h*h);
}


inline double a(double v, double nu, double h, double r) {
    return r*(-v*h/2.0+nu);
}

inline double b(double v, double nu, double h, double r) {
    return r*(v*h/2.0+nu);
}


// compute rhs with u
void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double rr, double nu, double h) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd; // LHS coeffficients

    #pragma omp parallel
    #pragma omp for collapse(2)
    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            rhs[i*(n+1)+j] = (1+4.0*rr*nu)*u[i*(n+1)+j] - cc*u[(i-1)*(n+1)+j] - aa*u[i*(n+1)+(j-1)] - dd*u[(i+1)*(n+1)+j] - bb*u[i*(n+1)+(j+1)];
        }
    }
}

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double rr, double nu, double h) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd; // LHS coeffficients

    #pragma omp parallel
    #pragma omp for collapse(2)
    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            aa = a(v2[i*(n+1)+j],nu,h,rr);
            bb = b(v2[i*(n+1)+j],nu,h,rr);
            cc = a(v1[i*(n+1)+j],nu,h,rr);
            dd = b(v1[i*(n+1)+j],nu,h,rr);
            res[i*(n+1)+j] = rhs[i*(n+1)+j] - ((1-4.0*rr*nu)*u[i*(n+1)+j] + cc*u[(i-1)*(n+1)+j] + aa*u[i*(n+1)+(j-1)] + dd*u[(i+1)*(n+1)+j] + bb*u[i*(n+1)+(j+1)]);
        }
    }
}


double compute_norm(double *res, long n) {

    double tmp = 0.0;

    for (long i = 1; i < n; i++){
        for (long j = 1; j < n; j++) {
            tmp += res[i*(n+1)+j] * res[i*(n+1)+j];
        }
    }
    return sqrt(tmp);
}

void gauss_seidel(double *u, double *unew, double *rhs, long n, double *v1, double *v2, double rr, double nu, double h) {
    // k: time step (dt)
    // h: spatial discretization step (dx=dy)
    // r: dt/(2*dx*dx)

    double aa,bb,cc,dd; // LHS coeffficients

    #pragma omp parallel
    { 
        // GS iteration to update red points
        #pragma omp for collapse(2) nowait
        for (long i=1; i < n; i+=2) { // iterate through every odd row of red points (starting at 1)
                for (long j=1; j < n; j+=2) { // iterate through every other column of red points at each row
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    unew[i*(n+1)+j] = (rhs[i*(n+1)+j] + cc*u[(i-1)*(n+1)+j] + aa*u[i*(n+1)+(j-1)] + dd*u[(i+1)*(n+1)+j] + bb*u[i*(n+1)+(j+1)])/(1-4.0*rr*nu);
                }
                
        }

        #pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of red points (starting at 2)
                for (long j=2; j < n; j+=2) {
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    unew[i*(n+1)+j] = (rhs[i*(n+1)+j] + cc*u[(i-1)*(n+1)+j] + aa*u[i*(n+1)+(j-1)] + dd*u[(i+1)*(n+1)+j] + bb*u[i*(n+1)+(j+1)])/(1-4.0*rr*nu);
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
                    unew[i*(n+1)+j] = (rhs[i*(n+1)+j] + cc*unew[(i-1)*(n+1)+j] + aa*unew[i*(n+1)+(j-1)] + dd*unew[(i+1)*(n+1)+j] + bb*unew[i*(n+1)+(j+1)])/(1-4.0*rr*nu);
                }
        }

        #pragma omp for collapse(2)
        for (long i=2; i < n; i+=2) { // iterate through every even row of black points (starting at 1)
                for (long j=1; j < n; j+=2) {
                    aa = a(v2[i*(n+1)+j],nu,h,rr);
                    bb = b(v2[i*(n+1)+j],nu,h,rr);
                    cc = a(v1[i*(n+1)+j],nu,h,rr);
                    dd = b(v1[i*(n+1)+j],nu,h,rr);
                    unew[i*(n+1)+j] = (rhs[i*(n+1)+j] + cc*unew[(i-1)*(n+1)+j] + aa*unew[i*(n+1)+(j-1)] + dd*unew[(i+1)*(n+1)+j] + bb*unew[i*(n+1)+(j+1)])/(1-4.0*rr*nu);
                }
        }
    }
        double* utmp = u;
        u = unew;
        unew = utmp;


}