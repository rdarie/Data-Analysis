#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=2:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=48
#SBATCH --tasks=48
#SBATCH --tasks-per-node=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J spike_sort_constructor
#SBATCH --array=2,3,4

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-spike_sort_constructor.stdout
#SBATCH -e ../batch_logs/%j-%a-spike_sort_constructor.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
# EXP="exp201901221000"
EXP="exp201901070700"
module load mpi
srun --mpi=pmi2 ./tridesclousCCV.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPreprocess
