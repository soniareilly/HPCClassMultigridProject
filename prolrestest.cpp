#include <stdio.h>
#include <math.h>
#include <string.h>

// u is row major

void prolongation(double* up, double* u, int n){
    // up is the output, size 2n+1 x 2n+1, u is the input, size n+1 x n+1
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            up[2*i*(2*n+1) + 2*j] = u[i*(n+1) + j];
            up[(2*i+1)*(2*n+1) + 2*j] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j])/2;
            up[2*i*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[i*(n+1) + j+1])/2;
            up[(2*i+1)*(2*n+1) + 2*j+1] = (u[i*(n+1) + j] + u[(i+1)*(n+1) + j] + u[i*(n+1) + j+1] + u[(i+1)*(n+1) + j+1])/4;
        }
    }
    // right and bottom borders
    for (int i = 0; i < n; i++){
        // right border
        up[2*i*(2*n+1) + 2*n] = u[i*(n+1) + n];
        up[(2*i+1)*(2*n+1) + 2*n] = (u[i*(n+1) + n] + u[(i+1)*(n+1) + n])/2;
        // bottom border
        up[2*n*(2*n+1) + 2*i] = u[n*(n+1) + i];
        up[2*n*(2*n+1) + 2*i+1] = (u[n*(n+1) + i] + u[n*(n+1) + i+1])/2;
    }
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
            u[i*(n/2+1)+j] = up[2*i*(n+1)+2*j];
        }
    }
}

int main(){
    int N = 5;
    
    double *up, *u, *up2;
    up = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );
    u = (double*) malloc ( sizeof(double) * (2*N+1)*(2*N+1) );
    up2 = (double*) malloc ( sizeof(double) * (N+1)*(N+1) );

    #pragma omp parallel num_threads(4)
    {
    
    #pragma omp single
    {

    // Initialize up
    #pragma omp for
    for (int i = 0; i < N+1; i++){
        for(int j = 0; j < N+1; j++){
            up[i*(N+1)+j] = i+j;
        }
    }

    printf("Original matrix\n");
    for (int i = 0; i < N+1; ++i)
    {   
        for (int j = 0; j < N+1; ++j)
        {
            printf("%.1f\t",up[i*(N+1)+j]);
        }
        printf("\n");
    }
    prolongation(u, up, N);
    printf("Prolongated matrix\n");
    for (int i = 0; i < 2*N+1; ++i)
    {   
        for (int j = 0; j < 2*N+1; ++j)
        {
            printf("%.1f\t",u[i*(2*N+1)+j]);
        }
        printf("\n");
    }
    restriction(up2, u, 2*N);
    printf("Restriction matrix\n");
    for (int i = 0; i < N+1; ++i)
    {   
        for (int j = 0; j < N+1; ++j)
        {
            printf("%.1f\t",up2[i*(N+1)+j]);
        }
        printf("\n");
    }
    }
    }

    free(up);
    free(u);
    free(up2);
}