/*  2D Multigrid Implementation for Advection-Diffusion using OpenMP
 *  2nd order finite difference spatial discretization, Crank-Nicholson time discretization
 *  (C) Tanya Wang, Sonia Reilly, Nolan Reilly
 *  Spring 2022 High Performance Computing Final Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include<omp.h> 
#include "gs.h" 

#define PI 3.1415926535897932

// inner loop of multigrid, computes one V or W-cycle and recurses
void mg_inner(double** u, double** rhs, 
              double** v1, double** v2,
              double* tmp, double dx, int n, 
              int lvl, int maxlvl, 
              int shape, double dt, double nu)
{
/*
u[maxlvl][(n+1)*(n+1)]   - tower of arrays of u and its residual at successively 
                           coarsening grids
rhs[maxlvl][(n+1)*(n+1)] - same thing, but for the right hand side of the linear system
v1[maxlvl][(n+1)*(n+1)]  - same for x component of velocity at all grids
v2[maxlvl][(n+1)*(n+1)]  - same for y component of velocity at all grids
tmp[(n+1)*(n+1)]         - Spare computation space
dx                       - grid spacing at current level
n                        - n+1 cells, counting ghost nodes, in each dimension (so fields
                           are (n+1)*(n+1) in length)
lvl                      - Current level in the multigrid recursion
maxlvl                   - Maximum multigrid recursion depth
shape                    - Shape of multigrid cycles (shape=1 --> V-cycle, 2 --> W-cycle)
dt                       - timestep
nu                       - diffusion parameter
*/

    int i, iter, sh;
    int NITER = 3;      //number of Gauss-Seidel iterations

    // helper variables
    double *ui = u[lvl]; double *ui1 = u[lvl+1];
    double *rhsi = rhs[lvl]; double *rhsi1 = rhs[lvl+1];
    double *v1i = v1[lvl];
    double *v2i = v2[lvl];
    int nnew = n/2;
    double dx2 = 2*dx;

    // Loop over shape -- shape == 1 is V-cycle, shape == 2 is W-cycle
    for (sh = 0; sh < shape; ++sh)
    {
        // at coarsest level, solve with Gauss-Seidel
        if (lvl == maxlvl-1){
            // Solve for du = A\r, here using Gauss-Seidel until fully converged
            // Replace with exact solve, time permitting
            double res_exact = 1.0; i = 0;
            // maxiter and tolerance are hard-coded for now
            while (i < 1000 && res_exact > 1e-5){
                gauss_seidel(ui, rhsi, n, v1i, v2i, dt, nu, dx);
                residual(tmp, ui, rhsi, n, v1i, v2i, dt, nu, dx);
                res_exact = compute_norm(tmp, n);
                i++;
            }
        // at any other level, smooth, restrict, recurse, prolong, update
        } else{
            // smoothing
            for (iter = 0; iter < NITER; ++iter)
            {
                gauss_seidel(ui, rhsi, n, v1i, v2i, dt, nu, dx);
            }
            residual(tmp, ui, rhsi, n, v1i, v2i, dt, nu, dx);
            // restrict residual to coarser level
            restriction(rhsi1, tmp, n);
            // set u[lvl+1] to 0
            for (i = 0; i < (nnew+1)*(nnew+1); ++i) ui1[i] = 0;
            // Recurse at coarser level
            mg_inner(u, rhs, v1, v2, tmp, dx2, nnew, lvl+1, maxlvl, shape, dt, nu);
            // Prolong solution back to fine level
            prolongation(tmp, ui1, nnew);
            // update u
            for (i = 0; i < (n+1)*(n+1); ++i) ui[i] += tmp[i];
            // smoothing
            for (iter = 0; iter < NITER; ++iter)
            {
                gauss_seidel(ui, rhsi, n, v1i, v2i, dt, nu, dx);
            }
        }
    }
    return;
}

const long MAX_CYCLE = 50;      // maximum number of V or W-cycles

// outer loop of multigrid, repeats V or W-cycle as many times as necessary for convergence
void mg_outer(double** utow, double** v1tow, double** v2tow, double** rhstow, 
              double* tmp, double nu, int maxlvl, int n, 
              double dt, double dx, double tol, int shape) {

    double res_norm, res0_norm;
    int iter;
    // calculate initial residual
    residual(tmp, utow[0], rhstow[0], n, v1tow[0], v2tow[0], dt, nu, dx);
    res_norm = res0_norm = compute_norm(tmp, n);

    // perform V or W-cycles until convergence
    for (iter = 0; iter < MAX_CYCLE && res_norm/res0_norm > tol; iter++) { 
        // V or W-cycle
        mg_inner(utow, rhstow, v1tow, v2tow, tmp, dx, n, 0, maxlvl, shape, dt, nu);
        // compute new residual
        residual(tmp, utow[0], rhstow[0], n, v1tow[0], v2tow[0], dt, nu, dx);
        res_norm = compute_norm(tmp,n);
    }

    // test for non-convergence and print warning
    if (iter == MAX_CYCLE-1){
        printf("multigrid did not converge in %li cycles\n", MAX_CYCLE);
    }
}

// Step solution forward in time
// Outputs the solution uT after performing the timestepping starting with u0
void timestepper(double* uT, double* u0, double* v1, double* v2,
                double nu, int maxlvl, int n, double dt, double T, 
                double dx, double tol, int shape){
    // n is the dimension of u0, v1, v2, the finest n
    // ntr is the number of threads for omp

    // declare towers (pointers to arrays of pointers to successively coarsening arrays)
    double *utow[maxlvl]; 
    double *v1tow[maxlvl]; 
    double *v2tow[maxlvl]; 
    double *rhstow[maxlvl];
    
    // initialize top (finest) levels of towers
    // utow[0] = u0; v1tow[0] = v1; v2tow[0] = v2; 
    utow[0]   = (double*) malloc((n+1)*(n+1)*sizeof(double));
    v1tow[0]  = (double*) malloc((n+1)*(n+1)*sizeof(double));
    v2tow[0]  = (double*) malloc((n+1)*(n+1)*sizeof(double));
    rhstow[0] = (double*) malloc((n+1)*(n+1)*sizeof(double));
    
    memcpy(utow[0], u0, (n+1)*(n+1)*sizeof(double));
    memcpy(v1tow[0], v1, (n+1)*(n+1)*sizeof(double));
    memcpy(v2tow[0], v2, (n+1)*(n+1)*sizeof(double));

    // define/initialize lower levels
    int ni = n+1;
    for (int i = 1; i < maxlvl; i++){
        ni = (n>>1) + 1;
        // lower levels of utow are unitialized
        utow[i] = (double*) malloc(ni*ni* sizeof(double));
        // vtow has the progressive restrictions
        v1tow[i] = (double*) malloc(ni*ni* sizeof(double));
        restriction(v1tow[i],v1tow[i-1],ni-1);
        v2tow[i] = (double*) malloc(ni*ni* sizeof(double));
        restriction(v2tow[i],v2tow[i-1],ni-1);
        // lower levels of rhs are just 0s
        rhstow[i] = (double*) calloc(ni*ni, sizeof(double));
    }
    // initialize workspace variable
    double *tmp = (double *) malloc((n+1)*(n+1)*sizeof(double));

    // iterate in time
    for (int iter = 0; iter < (int) (T/dt); iter++){
        // update rhs of the linear system
        compute_rhs(rhstow[0], utow[0], n, v1tow[0], v2tow[0], dt, nu, dx);
        // solve the linear system
        mg_outer(utow, v1tow, v2tow, rhstow, tmp, nu, maxlvl, n, dt, dx, tol, shape);
        // print every 100 timestep number
        //if (iter%10 == 0) printf("Timestep number %i\n", iter);
    }

    // update uT
    memcpy(uT, utow[0], (n+1)*(n+1)*sizeof(double));

    free(tmp);
    // free each level of the towers
    for (int i = 0; i < maxlvl; ++i)
    {
        free(utow[i]);
        free(rhstow[i]);
        free(v1tow[i]);
        free(v2tow[i]);
    }
}

int main(){

//for (int N = 32; N <= 4096; N<<=1){
    // define N and calculate maxlvl
    int N = 256;                    // Finest grid size. MUST BE A POWER OF 2
    int maxlvl = int(log2(N))-4;    // Levels of multigrid. n = 16 is solved exactly
    double dx = 1.0/N;
    
    // allocate arrays
    double *uTref, *uTomp, *uTcuda, *u0, *v1, *v2;
    uTref = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    uTomp = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    uTcuda = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    u0 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    v1 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    v2 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );

    // define parameters of u0 and v1,v2
    double x0 = 0.2, y0 = 0.4;
    double sigma = 100.0;
    double kx = 1.0*PI;
    double ky = 1.0*PI;

    int i,j;

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
    // set the boundary of u0 to 0
    for (i = 0; i < N; ++i)
    {
        u0[i] = 0.0;
        u0[i*(N+1)+N] = 0.0;
        u0[N*(N+1)+i+1] = 0.0;
        u0[i*(N+1)] = 0.0;
    }
    // initialize diffusion parameter nu
    double nu = -4*1e-4;            // must be negative because of how we write the equation

    // call timestepper
    double dt = dx/10;              // timestep dt depends on dx to satisfy CFL
    double T  = 100*dt;             // end time of simulation
    double tol = 1e-6;              // relative tolerance for mg_outer convergence
    int shape = 1;                  // V-cycle or W-cycle (here, V-cycle)

    // Referenced computation on CPU with 1 thread
    double tt = omp_get_wtime();
    timestepper(uTref, u0, v1, v2, nu, maxlvl, N, dt, T, dx, tol, shape);
    printf("\nCPU (1 thread for reference) time, N = %i: %f s\n", N, omp_get_wtime()-tt);
    
    // Computation on CPU with 16 threads (OMP)
    double error = 0;
    tt = omp_get_wtime();
    int ntr = 8; // number of threads
    #pragma omp parallel num_threads(ntr) 
    {
        #pragma omp single
        {
        timestepper(uTomp, u0, v1, v2, nu, maxlvl, N, dt, T, dx, tol, shape);
        }
    }
    printf("\nCPU with OMP (%d threads) time, N = %i: %f s\n", ntr, N, omp_get_wtime()-tt);
    // compute error wrt the referenced solution
     for (i = 0; i < N+1; i++){
        for (j = 0; j < N+1; j++){
            error += fabs(uTomp[i*(N+1)+j]-uTref[i*(N+1)+j]);
        }
    }
    printf("Error (compared to the referenced solution) = %10e\n", error);

    // Print final uT to file
    FILE *f = fopen("uT.txt","w");
    for (i = 0; i < N+1; i++){
        for (j = 0; j < N+1; j++){
            fprintf(f, "%d\t%d\t%f\n", i, j, uTref[i*(N+1)+j]);
        }
    }
    fclose(f);

    // Print final uTomp to file
    f = fopen("uTomp.txt","w");
    for (i = 0; i < N+1; i++){
        for (j = 0; j < N+1; j++){
            fprintf(f, "%d\t%d\t%f\n", i, j, uTomp[i*(N+1)+j]);
        }
    }
    fclose(f);

    free(uTref);
    free(uTomp);
    free(uTcuda);
    free(u0);
    free(v1);
    free(v2);
//}
}
