// function headers for computing residuals

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr);

double compute_norm(double *res, long n, int ntr);

// function headers for Gauss-Seidel
// void gauss_seidel(double *unew, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h);
void gauss_seidel(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr);
void gauss_seidel2(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h, int ntr);

// function header to compute rhs based on u
void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double k, double nu, double h, int ntr);

// function headers for prolongation and restriction
void prolongation(double* up, double* u, int n, int ntr);
void restriction(double* u, double* up, int n, int ntr);