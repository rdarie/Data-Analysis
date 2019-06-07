#!/bin/bash

# 04: Run peeler
# Request an hour of runtime:
#SBATCH --time=2:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=48
#SBATCH --tasks=48
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J spike_sort_peeler
#SBATCH --array=1,2,3

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-spike_sort_peeler.stdout
#SBATCH -e ../batch_logs/%j-%a-spike_sort_peeler.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
module load mpi
srun --mpi=pmi2 ./tridesclousCCV.py --trialIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --purgePeeler --batchPeel --makeCoarseNeoBlock
