// Implementation of Gauss-Seidel method for 2D Poisson
// I removed the convergence test, since it's not needed to run for a fixed number of iterations,
// and it slows down the code significantly.

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "utils.h"
#define N_thr 16

void GaussSeidel( long n, long maxiter, double *f, double *u) {
    int k = 0;
    int i; int j;
    while (k < maxiter) {
        // UPDATING RED POINTS
        // here we're assuming top left is red
        // interior
        #pragma omp parallel for num_threads(N_thr) private(i,j)
        for (i = 1; i < n-1; i++) {
    	    for (j = 2-i%2; j < n-1; j+=2) {
                u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
			}
        }
        // top boundary
        i = 0;
        for (j = 2-i%2; j < n-1; j+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i+1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
        }
        // bottom boundary
        i = n-1;
        for (j = 2-i%2; j < n-1; j+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
        }
        // left boundary
        j = 0;
        for (i = 2-j%2; i < n-1; i+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j+1])/4;
        }
        // right boundary
        j = n-1;
        for (i = 2-j%2; i < n-1; i+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j-1])/4;
        }
        // top left corner
        u[0] = (f[0]/n/n+u[n]+u[1])/4;
        if ((n-1)%2 == 0) {
            // top right corner
            u[n-1] = (f[n-1]/n/n+u[n+n-1]+u[n-2])/4;
            // bottom left corner
            u[(n-1)*n] = (f[(n-1)*n]/n/n+u[(n-2)*n]+u[(n-1)*n+1])/4;
        }
        // bottom right corner
        u[(n-1)*n+n-1] = (f[(n-1)*n+n-1]/n/n+u[(n-2)*n+n-1]+u[(n-1)*n+n-2])/4;


        // UPDATING BLACK POINTS
        // interior
        #pragma omp parallel for num_threads(N_thr) private(i,j)
        for (i = 1; i < n-1; i++) {
    	    for (j = 1+i%2; j < n-1; j+=2) {
                u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
			}
        }
        // top boundary
        i = 0;
        for (j = 1+i%2; j < n-1; j+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i+1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
        }
        // bottom boundary
        i = n-1;
        for (j = 1+i%2; j < n-1; j+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[i*n+j-1]+u[i*n+j+1])/4;
        }
        // left boundary
        j = 0;
        for (i = 1+j%2; i < n-1; i+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j+1])/4;
        }
        // right boundary
        j = n-1;
        for (i = 1+j%2; i < n-1; i+=2) {
            u[i*n+j] = (f[i*n+j]/n/n+u[(i-1)*n+j]+u[(i+1)*n+j]+u[i*n+j-1])/4;
        }
        if ((n-1)%2 == 1) {
            // top right corner
            u[n-1] = (f[n-1]/n/n+u[n+n-1]+u[n-2])/4;
            // bottom left corner
            u[(n-1)*n] = (f[(n-1)*n]/n/n+u[(n-2)*n]+u[(n-1)*n+1])/4;
        }

        // Print u at the middle point
        //printf("u[n/2,n/2]: %10g\n", u[n*n/2+n/2]);

        // update iteration count
	    k++;
    }
}

int main(int argc, char** argv) {
    long ns[] = {30, 100, 300, 1000, 3000};
    long maxiter = 100;
    // loop over sizes of array
    for (int i = 0; i < 5; i++){
        long n = ns[i];
        double* f = (double*) malloc(n*n * sizeof(double)); 		// n
        double* u = (double*) malloc(n*n * sizeof(double)); 		// n

	    // Initialize f,u
        for (int i = 0; i < n*n; i++) f[i] = 1;
        for (int i = 0; i < n*n; i++) u[i] = 0;

        Timer t;
        t.tic();
        GaussSeidel(n, maxiter, f, u);
        double time = t.toc();
        printf("Time to run Gauss-Seidel with N = %10ld: %10f\n", n, time);

        free(u);
        free(f);
    }

    return 0;
}
