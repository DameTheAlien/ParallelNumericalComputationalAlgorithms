#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*
 * Damian Franco
 * CS-575
 * Final Project - Gaussian Elimination
 *
 * This program creates one randomized matrix A and a randomized
 * vector b and solves a linear system Ax = b for vector x with
 * Gaussian Elimination and gets the approximate time to compute.
 * There is a serial version and parallel version of Gaussian
 * Elimination below.
 */

// Performs serial gaussian elimination
void ge_serial(double **A, double *b, double *x, int n) {
    // Forward elimination
    for (int k = 0; k < n-1; k++) {
        for (int i = k+1; i < n; i++) {
            double xmult = A[i][k] / A[k][k];
            for (int j = k; j < n; j++) {
                A[i][j] = A[i][j] - xmult * A[k][j];
            }
            b[i] = b[i] - xmult * b[k];
        }
    }

    // Backward substitution
    x[n-1] = b[n-1] / A[n-1][n-1];
    for (int i = n-2; i >= 0; i--) {
        double s = b[i];
        for (int j = i+1; j < n; j++) {
            s = s - A[i][j] * x[j];
        }
        x[i] = s / A[i][i];
    }

    // Uncomment to print results
    // printf("Serial - Solution vector x:")
    // printf("[");
    // for (int i = 0; i < n; i++) {
    //     printf("%lf ", x[i]);
    // }
    // printf("]");
    // printf("\n");
}

// Performs parallel gaussian elimination
void ge_parallel(double **A, double *b, double *x, int n, int rank, int size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Split the rows of the matrix among different processes
    int chunk_size = n / size;
    int start = rank * chunk_size;
    int end = start + chunk_size;
    if (rank == size-1) {
        end = n;
    }

    // Forward elimination
    for (int k = 0; k < n-1; k++) {
        // Broadcast the pivot row to all processes
        double pivot_row[n];
        if (rank == k / chunk_size) {
            for (int j = 0; j < n; j++) {
                pivot_row[j] = A[k][j];
            }
        }
        MPI_Bcast(pivot_row, n, MPI_DOUBLE, k / chunk_size, MPI_COMM_WORLD);

        // Eliminate the rows in parallel
        for (int i = start; i < end; i++) {
            if (i <= k) {
                continue;
            }
            double xmult = A[i][k] / pivot_row[k];
            for (int j = k; j < n; j++) {
                A[i][j] = A[i][j] - xmult * pivot_row[j];
            }
            b[i] = b[i] - xmult * b[k];
        }

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Backward substitution
    for (int i = end-1; i >= start; i--) {
        double s = b[i];
        for (int j = i+1; j < n; j++) {
            s = s - A[i][j] * x[j];
        }
        x[i] = s / A[i][i];
    }

    // Gather the solutions from all processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x, chunk_size, MPI_DOUBLE, MPI_COMM_WORLD);

    // Uncomment to print results
    // if (rank == 0) {
    //     printf("Parallel - Solution vector x:")
    //     printf("[");
    //     for (int i = 0; i < n; i++) {
    //         printf("%lf ", x[i]);
    //     }
    //     printf("]");
    //     printf("\n");
    // }
}

int main(int argc, char *argv[]) {
    int n = 1000;
    double beforeGE_serial, afterGE_serial;
    double beforeGE_parallel, afterGE_parallel;

    // For double matrix and vectors
    double **A = new double*[n];
    double *b = new double[n];
    double *x = new double[n];
    for (size_t i = 0; i < n; i++) {
       A[i] = new double[n];
    }

    // Fill matrix A and vector b with random double values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 1000000) + 1;
        }
        b[i] = (double)(rand() % 1000000) + 1;
    }

    printf("Matrices initialized, running algorithms:");

    // Gaussian Elimination Serial multiplication run
    beforeGE_serial = MPI_Wtime();
    ge_serial(A, b, x, n);
    afterGE_serial = MPI_Wtime();
    double elapsedTime_serial = afterGE_serial - beforeGE_serial;

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    // Gaussian Elimination Parallel multiplication run
    beforeGE_parallel = MPI_Wtime();
    ge_parallel(A, b, x, n, rank, size);
    afterGE_parallel = MPI_Wtime();
    MPI_Finalize();
    double elapsedTime_parallel = afterGE_parallel - beforeGE_parallel;

    printf("\n");
    printf("Time taken to complete power method (serial): %7.16lf secs\n", elapsedTime_serial);
    printf("Time taken to complete power method (parallel): %7.16lf secs\n", elapsedTime_parallel);

    for (size_t i = n; i > 0; ) {
       delete[] A[--i];
    }
    delete[] A;
    delete[] b;
    delete[] x;

    return 0;
}
