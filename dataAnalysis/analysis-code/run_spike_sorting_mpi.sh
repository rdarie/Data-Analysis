#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=2:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=48
#SBATCH --tasks=48
#SBATCH --tasks-per-node=1
#SBATCH --mem=16G

# Specify a job name:
#SBATCH -J spike_sort
#SBATCH --array=5

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-spike_sort-o.out
#SBATCH -e ../batch_logs/%j-%a-spike_sort-e.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
module load mpi
srun --mpi=pmi2 ./tridesclousCCV.py --trialIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPeel --purgePeeler
