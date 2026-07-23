#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*
 * Damian Franco
 * CS-575
 * Final Project - Eigenvalue Approximation with Power Method
 *
 * This program creates one randomized matrix A and computes the
 * all of the eigenvalue of the matrix with the Power Method
 * approach and gets the approximate time to compute.
 * There is a serial version and parallel version of the Power
 * method eigenvalue approximation below.
 */

// Performs eigenvalue approximation with Power method in serial
void power_serial(double **A, int n, int iterations, double tol) {
// Allocate memory for x and y
    double *x = (double*) malloc(n * sizeof(double));
    double *y = (double*) malloc(n * sizeof(double));
    // Intial eigenvalues
    double lambda = 1.0, lambda_old = 0.0;

    // Initialize eigenvector to [1, 1, ..., 1]^T
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }

    // Iterate until convergence or iter is reached
    for (int iter = 0; iter < iterations && fabs(lambda - lambda_old) > tol; iter++) {
        lambda_old = lambda;

        // Compute y = A*x
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
            for (int j = 0; j < n; j++) {
                y[i] += A[i][j] * x[j];
            }
        }

        // Compute lambda = y^T*x / x^T*x
        lambda = 0.0;
        double norm_x = 0.0;
        for (int i = 0; i < n; i++) {
            lambda += y[i] * x[i];
            norm_x += x[i] * x[i];
        }
        lambda /= norm_x;

        // Normalize x to unit length
        double norm_x_new = 0.0;
        for (int i = 0; i < n; i++) {
            x[i] = y[i] / lambda;
            norm_x_new += x[i] * x[i];
        }
        norm_x_new = sqrt(norm_x_new);
        for (int i = 0; i < n; i++) {
            x[i] /= norm_x_new;
        }
    }

    // Uncomment to print results
    // printf("Serial - Eigenvalue: %f\n", lambda);
    // printf("Serial - Eigenvector: [");
    // for (int i = 0; i < n; i++) {
    //     printf("%lf ", x[i]);
    // }
    // printf("]\n");

    // Free memory
    free(x);
    free(y);
}

// Performs eigenvalue approximation with Power method in parallel
void power_parallel(double **A, int n, int iterations, double tol, int rank, int size) {
    // MPI intialization cont.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Allocate memory for x and y
    double *x = (double*) malloc(n * sizeof(double));
    double *y = (double*) malloc(n * sizeof(double));
    // Intial eigenvalues
    double lambda = 1.0, lambda_old = 0.0;

    // Initialize eigenvector to [1, 1, ..., 1]^T
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
    }

    // Divide the work among the MPI processes
    int chunk_size = n / size;
    int start_index = rank * chunk_size;
    int end_index = (rank == size - 1) ? n : (rank + 1) * chunk_size;

    // Iterate until convergence or max_iter is reached
    for (int iter = 0; iter < max_iter && fabs(lambda - lambda_old) > eps; iter++) {
        lambda_old = lambda;

        // Compute y = A*x
        for (int i = start_index; i < end_index; i++) {
            y[i] = 0.0;
            for (int j = 0; j < n; j++) {
                y[i] += A[i][j] * x[j];
            }
        }

        // Compute lambda = y^T*x / x^T*x
        double local_lambda = 0.0;
        double norm_x = 0.0;
        for (int i = start_index; i < end_index; i++) {
            local_lambda += y[i] * x[i];
            norm_x += x[i] * x[i];
        }

        // Combine the local lambdas and norms from each process using MPI_Allreduce
        MPI_Allreduce(&local_lambda, &lambda, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&norm_x, &norm_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        lambda /= norm_x;

        // Update x
        for (int i = start_index; i < end_index; i++) {
            x[i] = y[i] / lambda;
        }

        // Synchronize x across processes using MPI_Allgather
        MPI_Allgather(MPI_IN_PLACE, chunk_size, MPI_DOUBLE, x + start_index, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Uncomment to print results
    // if (rank == 0) {
    //     printf("Parallel - Eigenvalue: %f\n", lambda);
    //     printf("Parallel - Eigenvector: [");
    //     for (int i = 0; i < n; i++) {
    //         printf("%lf ", x[i]);
    //     }
    //     printf("]\n");
    // }

    // Free memory
    free(x);
    free(y);
}

int main(int argc, char *argv[]) {
    int n = 8000;
    double beforeEval_serial, afterEval_serial;
    double beforeEval_parallel, afterEval_parallel;

    // Set up matrix A and eigenvalue list and eigenvector matrix
    double **A = new double*[n];

    for (size_t i = 0; i < n; i++) {
       A[i] = new double[n];
    }
    // Fill matrix A and vector b with random double values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 1000000) + 1;
        }
    }

    printf("Matrices initialized, running algorithms:");

    // Power method Serial run
    beforeEval_serial = MPI_Wtime();
    power_serial(A, n, 1000, 1e-6);
    afterEval_serial = MPI_Wtime();
    double elapsedTime_serial = afterEval_serial - beforeEval_serial;

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    // Power method Parallel run
    beforeEval_parallel = MPI_Wtime();
    power_parallel(A, n, 1000, 1e-6, rank, size);
    afterEval_parallel = MPI_Wtime();
    MPI_Finalize();
    double elapsedTime_parallel = afterEval_parallel - beforeEval_parallel;

    printf("\n");
    printf("Time taken to complete power method (serial): %7.16lf secs\n", elapsedTime_serial);
    printf("Time taken to complete power method (parallel): %7.16lf secs\n", elapsedTime_parallel);

    for (size_t i = n; i > 0; ) {
       delete[] A[--i];
    }
    delete[] A;

    return 0;
}
