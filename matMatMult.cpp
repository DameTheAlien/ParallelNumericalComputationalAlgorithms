#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*
 * Damian Franco
 * CS-575
 * Final Project - Matrix-Matrix Multiplication
 *
 * This program creates two randomized matrices A and B
 * and performs matrix by matrix multiplication and gets the
 * approximate time to compute. There is a serial version and
 * parallel version of matrix-matrix multiplication below.
 */

// Performs dense serial matrix by matrix multiplication
void matmat_serial(double **A, double **B, double **C, int n) {
    // Matrix C is an empty nxn matrix for resulting matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Uncomment to print results
    // printf("Serial - Resulting C matrix:")
    // printf("[")
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ", C[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("]")
}

// Performs dense parallel matrix by matrix multiplication
void matmat_parallel(double **A, double **B, double **C, int n, int rank, int size) {
    // Matrix C is an empty nxn matrix for resulting matrix
    // MPI intialization
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Set up number of rows for each process
    int rows_per_proc = n / size;
    int leftover_rows = n % size;

    // Initialize the local portion of the resulting matrix, avoid using local for A and B
    double **C_local = new double*[rows_per_proc];
    for (int i = 0; i < rows_per_proc; i++) {
        C_local[i] = new double[n];
        for (int j = 0; j < n; j++) {
            C_local[i][j] = 0.0;
        }
    }

    // Compute the starting and ending row indices for this process
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (rank == size - 1) {
        end_row += leftover_rows;
    }

    // Perform the matrix multiplication locally
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C_local[i-start_row][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Combine the local portions of the resulting matrix
    MPI_Allreduce(MPI_IN_PLACE, &C_local[0][0], rows_per_proc*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Copy the local portion of the resulting matrix to the global matrix
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = C_local[i-start_row][j];
        }
    }

    // Uncomment to print results
    // if (rank == 0) {
    //     printf("Parallel - Resulting C matrix:")
    //     printf("[")
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             printf("%lf ", C[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("]")
    // }

    // Free the memory used by the local portion of the resulting matrix
    for (int i = 0; i < rows_per_proc; i++) {
        delete[] C_local[i];
    }
    delete[] C_local;
}


int main(int argc, char *argv[]) {
    int n = 100;
    double beforeMxM_serial, afterMxM_serial;
    double beforeMxM_parallel, afterMxM_parallel;

    // For double matrices
    double **A = new double*[n];
    double **B = new double*[n];
    double **C = new double*[n];
    for (size_t i = 0; i < n; i++) {
       A[i] = new double[n];
       B[i] = new double[n];
       C[i] = new double[n];
    }

    // Fill matrices with random double values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 1000000) + 1;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = (double)(rand() % 1000000) + 1;
        }
    }

    printf("Matrices initialized, running algorithms:");

    // Matrix x Matrix Serial multiplication run
    beforeMxM_serial = MPI_Wtime();
    matmat_serial(A, B, C, n);
    afterMxM_serial = MPI_Wtime();
    double elapsedTime_serial = afterMxM_serial - beforeMxM_serial;

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    // Matrix x Matrix Parallel multiplication run
    beforeMxM_parallel = MPI_Wtime();
    matmat_parallel(A, B, C, n, rank, size);
    afterMxM_parallel = MPI_Wtime();
    MPI_Finalize();
    double elapsedTime_parallel = afterMxM_parallel - beforeMxM_parallel;

    printf("\n");
    printf("Time taken to complete power method (serial): %7.16lf secs\n", elapsedTime_serial);
    printf("Time taken to complete power method (parallel): %7.16lf secs\n", elapsedTime_parallel);

    for (size_t i = n; i > 0; ) {
       delete[] C[--i];
       delete[] B[i];
       delete[] A[i];
    }
    delete[] C;
    delete[] B;
    delete[] A;

    return 0;
}
