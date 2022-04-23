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

void mg_inner(double** u, double** f, 
              double* tmp1, double* tmp2, 
              double dx, int n, 
              int lvl, int maxlvl, 
              int shape)
{
    // Inner function for multigrid solver
    int i, iter, sh;
    int NITER = 50; //number of Gauss-Seidel iterations
        // I don't know how many to use; experiment!

    ui = u[lvl];
    fi = f[lvl];
    int nnew = n/2;
    double dx2 = 2*dx;

    // Loop over shape -- shape == 1 is V-cycle, shape == 2 is W-cycle
    for (sh = 0; sh < shape; ++sh)
    {
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, fi, dx, n);
        }
        residual(tmp1, ui, fi, n, dx);       // tmp1 <- residual(u, f, n) - residual n
        restriction(tmp2, tmp1, n);      // tmp2 <- restriction(tmp2, n) - residual nnew
        if (lvl == maxlvl)
        {
            // Explicit solve for du = A\r
            exact_solve(tmp1, tmp2, nnew);  // tmp1 <- exact_solve(tmp2, nnew)
            for (i = 0; i < nnew; ++i)  tmp2[i] += tmp1[i];
        } else 
        {
            // Multigrid should output in 1st arg with same dimension as input
            mg_inner(u, f, tmp1, tmp2, dx2, nnew, lvl+1, maxlvl, shape); // r <- mg(stuff)
        }
        prolongation(tmp1, u[lvl+1], nnew);  // up <- prolongation(u, nnew)
        for (i = 0; i < n; ++i) ui[i] += tmp1[i];
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(ui, fi, dx, n);
        }
    }
    // Output should be in u[lvl]!
    return;
}