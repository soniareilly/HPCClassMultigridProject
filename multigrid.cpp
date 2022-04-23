#include <stdio.h>
#include <math.h>
//#include "prolres.h"  // Prolongation + restriction header
#include "gs.h"       // Gauss-Seidel header

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

void mg_inner(double** u, double** f, 
              double* tmp1, double* tmp2, 
              double dx, int n, 
              int lvl, int maxlvl, 
              int shape, double dt, double *v1, double *v2, double nu)
{
    // Inner function for multigrid solver
    int i, iter, sh;
    int NITER = 50; //number of Gauss-Seidel iterations
        // I don't know how many to use; experiment!
    
    // compute coefficient for LHS and RHS
    double rr = r(dx,dt);

    double *ui = u[lvl];
    double *fi = f[lvl];
    int nnew = n/2;
    double dx2 = 2*dx;

    double* unew = (double *) calloc(sizeof(double), (n+1)*(n+1));

    // Loop over shape -- shape == 1 is V-cycle, shape == 2 is W-cycle
    for (sh = 0; sh < shape; ++sh)
    {
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, unew, fi, n, v1, v2, rr, nu, dx);

        }
        residual(tmp1, ui, fi, n, v1, v2, rr, nu, dx);      // tmp1 <- residual(u, f, n) - residual n
        restriction(tmp2, tmp1, n);      // tmp2 <- restriction(tmp2, n) - residual nnew
        if (lvl == maxlvl)
        {
            // Explicit solve for du = A\r
            exact_solve(tmp1, tmp2, nnew);  // tmp1 <- exact_solve(tmp2, nnew)
            for (i = 0; i < nnew; ++i)  tmp2[i] += tmp1[i];
        } else 
        {
            // Multigrid should output in 1st arg with same dimension as input
            mg_inner(u, f, tmp1, tmp2, dx2, nnew, lvl+1, maxlvl, shape, dt, v1, v2, nu); // r <- mg(stuff)
        }
        prolongation(tmp1, u[lvl+1], nnew);  // up <- prolongation(u, nnew)
        for (i = 0; i < n; ++i) ui[i] += tmp1[i];
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, unew, fi, n, v1, v2, rr, nu, dx);
        }
    }
    free(unew);
    // Output should be in u[lvl]!
    return;
}

void timestepper(double* uT, double* u0, double* v1, double* v2, 
                double nu, int maxlvl, int n, double dt, double T){
    // Outputs the solution u after performing the timestepping starting with u0
    // n is the dimension of u0, v1, v2, the finest n
    // declare towers
    double *utow[maxlvl]; double *v1tow[maxlvl]; double *v2tow[maxlvl];
    // copy u0
    double* u = (double*) malloc((n+1)*(n+1)*sizeof(double));
    for (int i=0; i<n+1; i++){
        u[i] = u0[i];
    }
    // initialize top levels of towers
    utow[0] = u; v1tow[0] = v1; v2tow[0] = v2;
    for (int i = 1; i < maxlvl; i++){
        // lower levels are unitialized
        utow[i] = (double*) malloc(((n>>i)+1)*((n>>i)+1)* sizeof(double));
        v1tow[i] = (double*) malloc(((n>>i)+1)*((n>>i)+1)* sizeof(double));
        v2tow[i] = (double*) malloc(((n>>i)+1)*((n>>i)+1)* sizeof(double));
    }

    // iterate
    for (int iter = 0; iter < (int) T/dt; iter++){
        mg_outer(utow, v1tow, v2tow, nu, maxlvl, n, dt); // utow <- mg_outer(stuff)
    }
}

int main(){
    // define N and calculate maxlvl
    // define v and nu
    // initialize u to some function

    // call timestepper
}

// allocate v1, v2, u, rhs

// let Tanya initialize rhs

// for loop calls mg_outer
// pass in current u