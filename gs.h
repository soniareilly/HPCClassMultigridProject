// function headers for Gauss-Seidel and computing residuals

void residual(double *res, double *u, double *f, long n, double *v1, double *v2, double rr, double nu, double h);

void gauss_seidel(double *u, double *unew, double *f, long n, double *v1, double *v2, double rr, double nu, double h);

// function headers for computing LHS and RHS coefficients
double r(double h, double k);
double a(double v, double nu, double h, double r);
double b(double v, double nu, double h, double r);