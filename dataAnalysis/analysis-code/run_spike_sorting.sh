#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=2,3,4

# Specify a job name:
#SBATCH -J spike_sorting

# Specify an output file
#SBATCH -o ../batch_logs/sorting-%j-%a.out
#SBATCH -e ../batch_logs/sorting-%j-%a.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID
