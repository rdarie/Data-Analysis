#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=8:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=4
#SBATCH --tasks=8
#SBATCH --tasks-per-node=2
#SBATCH --mem=4G

# Specify a job name:
#SBATCH -J mpi_test

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-mpi_test-o.out
#SBATCH -e ../batch_logs/%j-%a-mpi_test-e.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
module load mpi
# module load python/3.5.2
srun --mpi=pmi2 ./test_mpi.py --trialIdx=15

