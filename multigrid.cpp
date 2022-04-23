#include <stdio.h>
#include <math.h>
//#include "prolres.h"  // Prolongation + restriction header
//#include "gs.h"       // Gauss-Seidel header

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
            u[i*(n/2+1)+j] = (up[(2*i-1)*(n+1)+2*j-1] + 2*up[(2*i-1)*(n+1)+2*j] + up[(2*i-1)*(n+1)+2*j+1])/16;
            u[i*(n/2+1)+j] += (2*up[2*i*(n+1)+2*j-1] + 4*up[2*i*(n+1)+2*j] + 2*up[2*i*(n+1)+2*j+1])/16;
            u[i*(n/2+1)+j] += (up[(2*i+1)*(n+1)+2*j-1] + 2*up[(2*i+1)*(n+1)+2*j] + up[(2*i+1)*(n+1)+2*j+1])/16;

        }
    }
}

void mg_inner(double* up, double* u, 
              double** f, double* r, 
              double dx, int n, 
              int lvl, int maxlvl, 
              int shape)
{
    // Inner function for multigrid solver
    int i, iter, sh;
    int NITER = 50; //number of Gauss-Seidel iterations
        // I don't know how many to use; experiment!

    // Loop over shape -- shape == 1 is V-cycle, shape == 2 is W-cycle
    for (sh = 0; sh < shape; ++sh)
    {
        restriction(u, up, n);      // u <- restriction(up, n)
        int nnew = n/2;
        double dx2 = 2*dx;
        double* fi = f[lvl];        // double** f holds function f on all grids
        residual(r, u, fi, nnew);   // r <- residual(u, fi, n)
        if (lvl == maxlvl)
        {
            // Explicit solve for du = A\r
            exact_solve(up, r, nnew);  // up <- exact_solve(r, nnew)
            for (i = 0; i < nnew; ++i)  u[i] += up[i];
        } else 
        {
            for (iter = 0; iter < NITER; ++iter)
            {
                gauss_seidel(u, fi, dx2, nnew);
            }
            // Multigrid should output in up a u of the same dimension as it was given
            mg_inner(u, up, f, r, dx2, nnew, lvl+1, maxlvl, shape); // u <- mg(stuff)
        }
        prolongation(up, u, nnew);  // up <- prolongation(u, nnew)
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(up, f[lvl-1], dx, n);
        }
        return;
        // Output should be in up!
    }
}