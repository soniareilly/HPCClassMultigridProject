// function headers for Gauss-Seidel and computing residuals

void residual(double *res, double *u, double *f, long n, double aa, double bb, double cc, double dd, double rr, double nu);

void gauss_seidel(double *u, double *unew, double *f, long n, double aa, double bb, double cc, double dd, double rr, double nu);

// function headers for computing LHS and RHS coefficients
double r(double h, double k);
double a(double v, double nu, double h, double r);
double b(double v, double nu, double h, double r);