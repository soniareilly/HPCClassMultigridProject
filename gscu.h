// function headers for CUDA version of multigrid

__global__ void square_ker(double* a, long n);
__global__ void gpucopy(double* dest, const double* source, long n);
__global__ void reduction_kernel(double* sum, const double* a, long N);
__global__ void prolongation(double* up, double* u, int n);
__global__ void restriction(double *u, double *up, int n);
__global__ void compute_rhs(double* rhs, double* u, int N, double* v1, double* v2, double dt, double nu, double dx);
__global__ void residual(double* r, double* u, double* rhs, int N, double* v1, double* v2, double dt, double nu, double dx);
__global__ void jacobi_kernel(double* u, double* rhs, int N, double* v1, double* v2, double dt, double nu, double dx);
__global__ void gs_ker(double* u, double* rhs, long N, double* v1, double* v2, double dt, double nu, double dx, int rb);
__global__ void gaussian_u0(double* u0, double x0, double y0, double sigma, int n, double dx);
__global__ void rotating_v(double* v1, double* v2, double kx, double ky, int n, double dx);

double compute_norm(double* a, int N);
void gauss_seidel(double* u, double* rhs, long N, double* v1, double* v2, double dt, double nu, double dx);