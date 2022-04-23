#include <stdio.h>
#include <math.h>
//#include "prolres.h"  // Prolongation + restriction header
//#include "gs.h"       // Gauss-Seidel header

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
        }
        // Multigrid should output in up a u of the same dimension as it was given
        mg_inner(u, up, f, r, dx2, nnew, lvl+1, maxlvl, shape); // u <- mg(stuff)
        prolongation(up, u, nnew);  // up <- prolongation(u, nnew)
        for (iter = 0; iter < NITER; ++iter)
        {
            gauss_seidel(up, f[lvl-1], dx, n);
        }
        return;
        // Output should be in up!
    }
}