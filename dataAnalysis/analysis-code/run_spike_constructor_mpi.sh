#!/bin/bash
# 01: Preprocess spikes
# Request an hour of runtime:
#SBATCH --time=1:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=32
#SBATCH --tasks=32
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J spike_sort_constructor
#SBATCH --array=1,2,3,4,5

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-spike_sort_constructor.stdout
#SBATCH -e ../batch_logs/%j-%a-spike_sort_constructor.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Run a command
# EXP="exp201901221000"
# EXP="exp201901070700"
EXP="exp201901271000"
module load mpi
srun --mpi=pmi2 ./tridesclousCCV.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --attemptMPI --batchPreprocess
