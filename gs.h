// function headers for computing residuals

void residual(double *res, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h);

double compute_norm(double *res, long n);

// function headers for Gauss-Seidel
// void gauss_seidel(double *unew, double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h);
void gauss_seidel(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h);
void gauss_seidel2(double *u, double *rhs, long n, double *v1, double *v2, double k, double nu, double h);

// function header to compute rhs based on u
void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double k, double nu, double h);

// function headers for prolongation and restriction
void prolongation(double* up, double* u, int n);
void restriction(double* u, double* up, int n);