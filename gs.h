// function headers for computing residuals

void residual(double *res, double *u, double *f, long n, double *v1, double *v2, double rr, double nu, double h);

double compute_norm(double *res, long n) ;

// function headers for Gauss-Seidel
void gauss_seidel(double *u, double *unew, double *f, long n, double *v1, double *v2, double rr, double nu, double h);

// function headers for computing LHS and RHS coefficients
double r(double h, double k);
double a(double v, double nu, double h, double r);
double b(double v, double nu, double h, double r);

// function header to compute rhs based on u
void compute_rhs(double *rhs, double *u, long n, double *v1, double *v2, double rr, double nu, double h);