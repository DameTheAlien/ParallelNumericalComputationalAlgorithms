#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*
 * Damian Franco
 * CS-575
 * Final Project - LU Decomposition with Doolittles Algorithm
 *
 * This program creates one randomized matrix A and computes the
 * LU decomposition of the matrix with Doolittles algorithm
 * approach and gets the approximate time to compute.
 * There is a serial version and parallel version of Doolittles
 * LU decomposition below.
 */

// Performs LU decomposition with Doolittles algorithm in serial
void doolittle_serial(double **A, double **L, double **U, int n) {
    // Allocate memory for L and U matrices
    for (int i = 0; i < n; i++) {
        L[i] = new double[n];
        U[i] = new double[n];
    }

    // Initialize L and U matrices
    for (int i = 0; i < n; i++) {
        L[i][i] = 1; // Diagonal elements of L are 1
        for (int j = 0; j < n; j++) {
            U[i][j] = 0;
        }
    }

    // Perform LU decomposition
    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j++) {
            double sum = 0;
            for (int p = 0; p < k; p++) {
                sum += L[k][p] * U[p][j];
            }
            U[k][j] = A[k][j] - sum;
        }

        for (int i = k + 1; i < n; i++) {
            double sum = 0;
            for (int p = 0; p < k; p++) {
                sum += L[i][p] * U[p][k];
            }
            L[i][k] = (A[i][k] - sum) / U[k][k];
        }
    }

    // Uncomment to print results
    // printf("Serial - L matrix:\n");
    // printf("[")
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ", L[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("]")
    // printf("\n");

    // printf("Serial - U matrix:\n");
    // printf("[")
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ", U[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("]")
    // printf("\n");
}

// Performs LU decomposition with Doolittles algorithm in parallel
void doolittle_parallel(double **A, double **L, double **U, int n, int rank, int size) {
    // MPI intialization cont.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Calculate the size of each block
    int block_size = n / size;

    // Allocate memory for local L and U matrices
    double **L_local = new double*[block_size];
    double **U_local = new double*[block_size];
    for (int i = 0; i < block_size; i++) {
        L_local[i] = new double[n];
        U_local[i] = new double[n];
    }

    // Initialize local L and U matrices
    for (int i = 0; i < block_size; i++) {
        L_local[i][i] = 1; // Diagonal elements of L are 1
        for (int j = 0; j < n; j++) {
            U_local[i][j] = 0;
        }
    }

    // Divide matrix A into blocks and distribute among processes
    double *A_block = new double[block_size * n];
    MPI_Scatter(A[0], block_size * n, MPI_DOUBLE, A_block, block_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform LU decomposition on local block
    for (int k = 0; k < block_size; k++) {
        for (int j = k; j < n; j++) {
            double sum = 0;
            for (int p = 0; p < k; p++) {
                sum += L_local[k][p] * U_local[p][j];
            }
            U_local[k][j] = A_block[k * n + j] - sum;
        }

        for (int i = k + 1; i < block_size; i++) {
            double sum = 0;
            for (int p = 0; p < k; p++) {
                sum += L_local[i][p] * U_local[p][k];
            }
            L_local[i][k] = (A_block[i * n + k] - sum) / U_local[k][k];
        }
    }

    // Communicate necessary information to update global L and U matrices
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            MPI_Send(U_local[0], block_size * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(L_local[0], block_size * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        } else if (rank == 0) {
            MPI_Recv(U[i * block_size], block_size * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(L[i * block_size], block_size * n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Uncomment to print results
    // if (rank == 0) {
    //     printf("Parallel - L matrix:\n");
    //     printf("[")
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             printf("%lf ", L[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("]")
    //     printf("\n");

    //     printf("Parallel - U matrix:\n");
    //     printf("[")
    //     for (int i = 0; i < n; i++) {
    //         for (int j = 0; j < n; j++) {
    //             printf("%lf ", U[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("]")
    //     printf("\n");   
    // }

    // Free memory for local L and U matrices
    for (int i = 0; i < block_size; i++) {
        delete[] L_local[i];
        delete[] U_local[i];
    }
    delete[] L_local;
    delete[] U_local;
    delete[] A_block;
}


int main(int argc, char *argv[]) {
    int n = 10;
    double beforeLU_serial, afterLU_serial;
    double beforeLU_parallel, afterLU_parallel;

    // For double matrices
    double **A = new double*[n];
    double **L = new double*[n];
    double **U = new double*[n];
    for (size_t i = 0; i < n; i++) {
       A[i] = new double[n];
    }

    // Fill matrix A with random double values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 1000000) + 1;
        }
    }

    printf("Matrices initialized, running algorithms:");

    // Doolittle Decomposition Serial run
    beforeLU_serial = MPI_Wtime();
    doolittle_serial(A, L, U, n);
    afterLU_serial = MPI_Wtime();
    double elapsedTime_serial = afterLU_serial - beforeLU_serial;

    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    // Doolittle Decomposition Parallel run
    beforeLU_parallel = MPI_Wtime();
    doolittle_parallel(A, L, U, n, rank, size);
    afterLU_parallel = MPI_Wtime();
    MPI_Finalize();
    double elapsedTime_parallel = afterLU_parallel - beforeLU_parallel;

    printf("\n");
    printf("Time taken to complete power method (serial): %7.16lf secs\n", elapsedTime_serial);
    printf("Time taken to complete power method (parallel): %7.16lf secs\n", elapsedTime_parallel);

    for (size_t i = n; i > 0; ) {
       delete[] U[--i];
       delete[] L[--i];
       delete[] A[i];
    }
    delete[] U;
    delete[] L;
    delete[] A;

    return 0;
}
