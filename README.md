# Parallelism in Numerical Computational Algorithms
[View the project report](./FinalReport_DamianFranco.pdf)

## Damian Franco
## CS-575

This directory contains:
  -  matmatMult.cpp
  -  gaussElim.cpp
  -  luDecomp.cpp
  -  evalCalc.cpp
  -  PBS Scripts

To run each program, please use the following operations below:

MatMat:
1) mpic++ -o matMatMult matMatMult.cpp
2) mpirun -n "DESIRED NUMBER OF PROCESSES" -N "DESIRED NUMBER OF MAXIMUM PROCESSES" ./matMatMult

Gaussian Elimination:
1) mpic++ -o gaussElim gaussElim.cpp
2) mpirun -n "DESIRED NUMBER OF PROCESSES" -N "DESIRED NUMBER OF MAXIMUM PROCESSES" ./gaussElim

LU Decomposition:
1) mpic++ -o luDecomp luDecomp.cpp
2) mpirun -n "DESIRED NUMBER OF PROCESSES" -N "DESIRED NUMBER OF MAXIMUM PROCESSES" ./luDecomp

Eigenvalues Calculation:
1) mpic++ -o evalCalc evalCalc.cpp
2) mpirun -n "DESIRED NUMBER OF PROCESSES" -N "DESIRED NUMBER OF MAXIMUM PROCESSES" ./evalCalc

To run each program on a High Performance Computing device/Supercomputer: Please submit a Slurm or PBS job with the choosen PBS script.

Each program has a different serial and parallel algorithm for its respective algorithm. Features such as dimension size or result printing to console are located in the source code for each file so if there must be a change in dimensions, please locate the variable n in the source code and change it to the desired dimension. The run time of each version of the algorithm will be printed alongside each other for quick comparision.
