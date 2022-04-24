#include <stdio.h>
#include <lapacke.h>

class Factor {
public:
    
private:
    int N;
    int KL;
    int KU;
    double* AB;
    int LDAB;
    int* IPIV;
    double* WORK;
}

Factor::Factor(int n, int kl, int ku)
{
    N = n;
    KL = kl;
    KU = ku;
    LDAB = 2*kl+ku+1;
    AB = (double*) malloc ( sizeof(double) * LDAB * N );
    IPIV = (int*) malloc ( sizeof(int) * N );
}

Factor::~Factor()
{
    free(AB);
    free(IPIV);
}

void factorize(Factor f)
{
    // This *should* modify the contents of f.AB and f.IPIV, but nothing else
    int INFO;
    LAPACKE_dgbtrf(f.N, f.N, f.KL, f.KU, f.AB, f.LDAB, f.IPIV, &INFO);
    if (INFO == 0) {
        return;
    } else if (INFO == f.N+1) {
        printf("Condition number infinite, DGBTRF unreliable\n");
    } else {
        printf("Matrix singular\n");
    }
}

void solve(Factor f, double* X, int NRHS, char TRANS)
{
    int INFO;
    LAPACKE_dgbtrs(TRANS, f.N, f.KL, f.KU, NRHS, f.AB, f.LDAB, f.IPIV, X, f.N, &INFO);
    if (INFO == 0) {
        return;
    } else {
        printf ("In DGBTRS, illegal argument #%i",-INFO);
    }
}