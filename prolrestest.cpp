#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

// u is row major

void prolongation(double* up, double* u, int n){
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            #pragma omp task
            {
            up[2*i*(2*n+1) + 2*j] = u[i*(n+1) + j];
            up[(2*i+1)*(2*n+1) + 2*j] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j])/2;
            up[2*i*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[i*(n+1) + j+1])/2;
            up[(2*i+1)*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j] + u[i*(n+1) + j+1] + u[(i+1)*(n+1) + j+1])/4;
            }
        }
    }
    // right and bottom borders
    for (int i = 0; i < n; i++){
        // right border
        #pragma omp task
        {
        up[2*i*(2*n+1) + 2*n] = u[i*(n+1) + n];
        up[(2*i+1)*(2*n+1) + 2*n] = (u[i*(n+1) + n] + u[(i+1)*(n+1) + n])/2;
        // bottom border
        up[2*n*(2*n+1) + 2*i] = u[n*(n+1) + i];
        up[2*n*(2*n+1) + 2*i+1] = (u[n*(n+1) + i] + u[n*(n+1) + i+1])/2;
        }
    }
    #pragma omp taskwait
    // bottom right corner
    up[(2*n)*(2*n+1) + 2*n] = u[n*(n+1) + n];
}

void restriction(double* u, double* up, int n){
    // u is the output, size n/2+1 x n/2+1, up is the input, size n+1 x n+1
    for (int i = 0; i < n/2+1; i++){
        for (int j = 0; j < n/2+1; j++){
            //u[i*(n/2+1)+j] = (up[(2*i-1)*(n+1)+2*j-1] + 2*up[(2*i-1)*(n+1)+2*j] + up[(2*i-1)*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (2*up[2*i*(n+1)+2*j-1] + 4*up[2*i*(n+1)+2*j] + 2*up[2*i*(n+1)+2*j+1])/16;
            //u[i*(n/2+1)+j] += (up[(2*i+1)*(n+1)+2*j-1] + 2*up[(2*i+1)*(n+1)+2*j] + up[(2*i+1)*(n+1)+2*j+1])/16;
            #pragma omp task
            u[i*(n/2+1)+j] = up[2*i*(n+1)+2*j];
        }
    }
    #pragma omp taskwait
}

int main(){
    int N = 5;
    
    double *up, *u, *up2;
    up = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    u = (double*) malloc ( sizeof(double) * (2*N+1)*(2*N+1) );
    up2 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );

    uomp = (double*) malloc ( sizeof(double) * (2*N+1)*(2*N+1) );
    up2omp = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );

    // Initialize up
    for (int i = 0; i < N+1; i++){
        for(int j = 0; j < N+1; j++){

            up[i*(N+1)+j] = i+j;

        }
    }

    double tt = omp_get_wtime();
    prolongation(u, up, N);
    // restriction(up2, u, 2*N);
    printf("\nCPU (1 thread) time: %f s\n", omp_get_wtime()-tt);

    int ntr = 16;
    tt = omp_get_wtime();
    #pragma omp parallel num_threads(ntr)
    {
    
    #pragma omp single // manager thread
    {
    int tid = omp_get_thread_num();
    if(tid == 0) printf("number of threads: %d\n", omp_get_num_threads());
    
    prolongation(uomp, up, N);
    // restriction(up2omp, u, 2*N);
    
    }
    }
    printf("\nCPU with OMP (%d threads) time: %f s\n", ntr, omp_get_wtime()-tt);
    // compute error wrt the referenced solution
    for (i = 0; i < N+1; i++){
        for (j = 0; j < N+1; j++){
            error += fabs(u[i*(N+1)+j]-uomp[i*(N+1)+j]);
            // error += fabs(up2[i*(N+1)+j]-up2omp[i*(N+1)+j]);
        }
    }
    printf("Error = %10e\n", error);


    // printf("Original matrix\n");
    // for (int i = 0; i < N+1; ++i)
    // {   
    //     for (int j = 0; j < N+1; ++j)
    //     {
    //         printf("%.1f\t",up[i*(N+1)+j]);
    //     }
    //     printf("\n");
    // }
    // prolongation(u, up, N);
    // printf("Prolongated matrix\n");
    // for (int i = 0; i < 2*N+1; ++i)
    // {   
    //     for (int j = 0; j < 2*N+1; ++j)
    //     {
    //         printf("%.1f\t",u[i*(2*N+1)+j]);
    //     }
    //     printf("\n");
    // }
    // restriction(up2, u, 2*N);
    // printf("Restriction matrix\n");
    // for (int i = 0; i < N+1; ++i)
    // {   
    //     for (int j = 0; j < N+1; ++j)
    //     {
    //         printf("%.1f\t",up2[i*(N+1)+j]);
    //     }
    //     printf("\n");
    // }

    free(up);
    free(u);
    free(up2);
}
