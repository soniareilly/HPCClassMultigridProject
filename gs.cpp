#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gs.h"

// functions for computing coefficients for LHS (and RHS)
double r(double h, double k) {
    return 1.0/2*k/(h*h);
}

double a(double v, double nu, double h, double r) {
    return r*(-v*h/2.0+nu);
}

double b(double v, double nu, double h, double r) {
    return r*(v*h/2.0+nu);
}

// compute rhs as f
// void(double *f, double *u)

void residual(double *res, double *u, double *f, long n, double aa, double bb, double cc, double dd, double rr, double nu) {
    // k: time step (dt)
    // h: spatial discretizaiton step (dx/dy)

    #pragma omp parallel
    #pragma omp for collapse(2)
    for (long i = 1; i <= n; i++){
        for (long j = 1; j <= n; j++) {
            res[i*(n+2)+j] = f[i*(n+2)+j] - ((1-4.0*rr*nu)*u[i*(n+2)+j] + cc*u[(i-1)*(n+2)+j] + aa*u[i*(n+2)+(j-1)] + dd*u[(i+1)*(n+2)+j] + bb*u[i*(n+2)+(j+1)]);
        }
    }
}

void gauss_seidel(double *u, double *unew, double *f, long n, double aa, double bb, double cc, double dd, double rr, double nu) {
    // k: time step (dt)
    // h: spatial discretizaiton step (dx/dy)

    #pragma omp parallel
    { 
        // GS iteration to update red points
        #pragma omp for collapse(2) nowait
        for (long i=1; i <= n; i+=2) { // iterate through every odd row of red points (starting at 1)
                for (long j=1; j <= n; j+=2) { // iterate through every other column of red points at each row
                    unew[i*(n+2)+j] = (f[i*(n+2)+j] + cc*u[(i-1)*(n+2)+j] + aa*u[i*(n+2)+(j-1)] + dd*u[(i+1)*(n+2)+j] + bb*u[i*(n+2)+(j+1)])/(1-4.0*rr*nu);
                }
                
        }

        #pragma omp for collapse(2)
        for (long i=2; i <= n; i+=2) { // iterate through every even row of red points (starting at 2)
                for (long j=2; j <= n; j+=2) {
                    unew[i*(n+2)+j] = (f[i*(n+2)+j] + cc*u[(i-1)*(n+2)+j] + aa*u[i*(n+2)+(j-1)] + dd*u[(i+1)*(n+2)+j] + bb*u[i*(n+2)+(j+1)])/(1-4.0*rr*nu);
                }
                
        }
	
        // GS iteration to update black points
        #pragma omp for collapse(2) nowait
        for (long i=1; i <= n; i+=2) { // iterate through every odd row of black points (starting at 2)
            for (long j=2; j <= n; j+=2) { // iterate through every other column of black points at each row
                    unew[i*(n+2)+j] = (f[i*(n+2)+j] + cc*unew[(i-1)*(n+2)+j] + aa*unew[i*(n+2)+(j-1)] + dd*unew[(i+1)*(n+2)+j] + bb*unew[i*(n+2)+(j+1)])/(1-4.0*rr*nu);
                }
        }

        #pragma omp for collapse(2)
        for (long i=2; i <= n; i+=2) { // iterate through every even row of black points (starting at 1)
                for (long j=1; j <= n; j+=2) {
                    unew[i*(n+2)+j] = (f[i*(n+2)+j] + cc*unew[(i-1)*(n+2)+j] + aa*unew[i*(n+2)+(j-1)] + dd*unew[(i+1)*(n+2)+j] + bb*unew[i*(n+2)+(j+1)])/(1-4.0*rr*nu);
                }
        }
    }
        double* utmp = u;
        u = unew;
        unew = utmp;


}