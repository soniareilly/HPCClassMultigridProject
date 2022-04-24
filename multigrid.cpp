#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gs.h"       // Gauss-Seidel header

#define PI 3.1415926535897932

// u is row major

void prolongation(double* up, double* u, int n){
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    for (int i = 0; i < n+1; i++){
        for (int j = 0; j < n+1; j++){
            up[2*i*(2*n+1) + 2*j] = u[i*(n+1) + j];
            up[(2*i+1)*(2*n+1) + 2*j] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j])/2;
            up[2*i*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[i*(n+1) + j+1])/2;
            up[(2*i+1)*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j] + u[i*(n+1) + j+1] + u[(i+1)*(n+1) + j+1])/4;
        }
    }
}

void restriction(double* u, double* up, int n){
    // u is the output, size n/2+1 x n/2+1, up is the input, size n+1 x n+1
    for (int i = 0; i < n/2+1; i++){
        for (int j = 0; j < n/2+1; j++){
            //u[i*(n/2+1)+j] = (up[(2*i-1)*(n+1)+2*j-1] + 2*up[(2*i-1)*(n+1)+2*j] + up[(2*i-1)*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (2*up[2*i*(n+1)+2*j-1] + 4*up[2*i*(n+1)+2*j] + 2*up[2*i*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (up[(2*i+1)*(n+1)+2*j-1] + 2*up[(2*i+1)*(n+1)+2*j] + up[(2*i+1)*(n+1)+2*j+1])/16;
            u[i*(n/2+1)+j] = up[2*i*(n+1)+2*j];
        }
    }
}

void mg_inner(double** u, double** rhs, 
              double** v1, double** v2,
              double* tmp1, double* tmp2, 
              double dx, int n, 
              int lvl, int maxlvl, 
              int shape, double dt, double nu)
{
/*
u[maxlvl][(n+1)*(n+1)]   - tower of arrays of u and its residual at successively 
                         coarsening grids
rhs[maxlvl][(n+1)*(n+1)] - same thing, but for the right hand side of the linear system
v1[maxlvl][(n+1)*(n+1)]  - same for x component of velocity at all grids
v2[maxlvl][(n+1)*(n+1)]  - same for y component of velocity at all grids
tmp1[(n+1)*(n+1)]        - Spare computation space, 1
tmp2[(n+1)*(n+1)]        - Spare computation space, 2
dx                       - grid spacing at current level
n                        - n+1 cells, counting ghost nodes, in each dimension (so fields
                           are (n+1)*(n+1) in length)
lvl                      - Current level in the multigrid recursion
maxlvl                   - Maximum multigrid recursion depth
shape                    - Shape of multigrid cycles (shape=1 --> V-cycle, 2 --> W-cycle)
dt                       - timestep
nu                       - diffusion parameter
*/

    // Inner function for multigrid solver
    int i, iter, sh;
    int NITER = 3; //number of Gauss-Seidel iterations
        // I don't know how many to use; experiment!


    double *ui = u[lvl];
    double *rhsi = rhs[lvl];
    double *v1i = v1[lvl];
    double *v2i = v2[lvl];
    int nnew = n/2;
    double dx2 = 2*dx;

    // Loop over shape -- shape == 1 is V-cycle, shape == 2 is W-cycle
    for (sh = 0; sh < shape; ++sh)
    {
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, tmp1, rhsi, n, v1i, v2i, dt, nu, dx);

        }
        residual(tmp1, ui, rhsi, n, v1i, v2i, dt, nu, dx);      // tmp1 <- residual(u, rhs, n) - residual n
        restriction(tmp2, tmp1, n);      // tmp2 <- restriction(tmp1, n) - residual nnew
        if (lvl == maxlvl-1)
        {
            // Explicit solve for du = A\r
            //exact_solve(tmp1, tmp2, nnew);  // tmp1 <- exact_solve(tmp2, nnew)
            double res_exact = 1.0; i = 0;
            while (i < 1000 && res_exact > 1e-5){
                gauss_seidel(tmp1, tmp2, rhsi, nnew, v1i, v2i, dt, nu, dx); // tmp1 <- gs(stuff)
                for (int j = 0; j < (nnew+1)*(nnew+1); j++) tmp2[j] = tmp1[j];
                residual(tmp1, tmp2, rhsi, nnew, v1i, v2i, dt, nu, dx);
                res_exact = compute_norm(tmp1, nnew);
                i++;
            }
            prolongation(tmp2, tmp1, nnew);  // up <- prolongation(u, nnew)
        } else 
        {
            // Multigrid should output in 1st arg with same dimension as input
            mg_inner(u, rhs, v1, v2, tmp1, tmp2, dx2, nnew, lvl+1, maxlvl, shape, dt, nu); // r <- mg(stuff)
            prolongation(tmp2, u[lvl+1], nnew);  // up <- prolongation(u, nnew)
        }
        for (i = 0; i < (n+1)*(n+1); ++i) ui[i] += tmp2[i];
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, tmp1, rhsi, n, v1i, v2i, dt, nu, dx);
        }
    }
    // Output should be in u[lvl]!
    return;
}

const long MAX_CYCLE = 50; // maximum number of v- or w-cycles

void mg_outer(double** utow, double** v1tow, double** v2tow, double** rhstow, 
              double* tmp1, double* tmp2,
              double nu, int maxlvl, int n, 
              double dt, double dx, double tol, int shape) {

    double res_norm, res0_norm; // residual norm and tolerance

    residual(tmp1, utow[0], rhstow[0], n, v1tow[0], v2tow[0], dt, nu, dx);
    res_norm = res0_norm = compute_norm(tmp1,n);
    printf("Initial Residual: %f\n", res0_norm);

    for (long iter = 0; iter < MAX_CYCLE && res_norm/res0_norm > tol; iter++) { // terminate when reach max iter or hit tolerance

        mg_inner(utow, rhstow, v1tow, v2tow, tmp1, tmp2, dx, n, 0, maxlvl, shape, dt, nu);
        // This is risky as tmp1 holds a slightly outdated residual
        //res_norm = compute_norm(tmp1,n); 
        // Perhaps better to recalculate
        residual(tmp1, utow[0], rhstow[0], n, v1tow[0], v2tow[0], dt, nu, dx);
        res_norm = compute_norm(tmp1,n);

        if (0 == (iter % 10)) {
            printf("[Iter %ld] Residual norm: %2.8f\n", iter, res_norm);
        }

    }

}

void timestepper(double* uT, double* u0, double* v1, double* v2,
                double nu, int maxlvl, int n, double dt, double T, 
                double dx, double tol, int shape){
    // Outputs the solution u after performing the timestepping starting with u0
    // n is the dimension of u0, v1, v2, the finest n
    // declare towers
    double *utow[maxlvl]; 
    double *v1tow[maxlvl]; 
    double *v2tow[maxlvl]; 
    double *rhstow[maxlvl];
    
    // initialize top levels of towers
    // utow[0] = u0; v1tow[0] = v1; v2tow[0] = v2; 
    utow[0] = (double*) malloc((n+1)*(n+1)*sizeof(double));
    v1tow[0] = (double*) malloc((n+1)*(n+1)*sizeof(double));
    v2tow[0] = (double*) malloc((n+1)*(n+1)*sizeof(double));
    rhstow[0] = (double*) malloc((n+1)*(n+1)*sizeof(double));
    
    memcpy(utow[0], u0, (n+1)*(n+1)*sizeof(double));
    memcpy(v1tow[0], v1, (n+1)*(n+1)*sizeof(double));
    memcpy(v2tow[0], v2, (n+1)*(n+1)*sizeof(double));

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

    double *tmp1, *tmp2;
    tmp1 = (double *) malloc((n+1)*(n+1)*sizeof(double));
    tmp2 = (double *) malloc((n+1)*(n+1)*sizeof(double));

    // iterate
    for (int iter = 0; iter < (int) (T/dt); iter++){
        compute_rhs(rhstow[0], utow[0], n+1, v1tow[0], v2tow[0], dt, nu, dx);
        mg_outer(utow, v1tow, v2tow, rhstow, tmp1, tmp2, nu, maxlvl, n, dt, dx, tol, shape); // utow <- mg_outer(stuff)
    }

    // update uT
    memcpy(uT, utow[0], (n+1)*(n+1)*sizeof(double));

    free(tmp1);
    free(tmp2);
    for (int i = 0; i < maxlvl; ++i)
    {
        free(utow[i]);
        free(rhstow[i]);
        free(v1tow[i]);
        free(v2tow[i]);
    }
}

int main(){
    // define N and calculate maxlvl
    // define v and nu
    // initialize u to some function
    int N = 256;
    double dx = 1.0/N;
    int maxlvl = 5; // n = 32 seems exactly solvable
    double nu = 1e-2; // chosen at random
    
    double *uT, *u0, *v1, *v2;
    uT = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    u0 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    v1 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    v2 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    int i,j;
    double kx = 4.0*PI;
    double ky = 4.0*PI;
    double x0 = 0.2, y0 = 0.4;
    double sigma = 100.0;
    for (i = 0; i < N+1; ++i)
    {
        for (j = 0; j < N+1; ++j)
        {
            // Let's hope that this Gaussian is sufficiently close to 0 at the edges...
            u0[i*(N+1)+j] =  exp(sigma*( (i*dx-x0)*(i*dx-x0) + (j*dx-y0)*(j*dx-y0) ));
            v1[i*(N+1)+j] = -ky*sin(kx*i*dx)*cos(ky*j*dx);
            v2[i*(N+1)+j] =  kx*cos(kx*i*dx)*sin(ky*j*dx);
        }
    }

    // call timestepper
    double dt = 0.001;  // Ah, we'll have to experiment with this one
    double T  = 1*dt;   // debug first, get ambitious later
    double tol = 1e-6;
    int shape = 1;      // V-cycles
    timestepper(uT, u0, v1, v2, nu, maxlvl, N, dt, T, dx, tol, shape);

    // Do something with output
    printf("uT[0] = %f\n", uT[0]);

    free(uT);
    free(u0);
    free(v1);
    free(v2);
}