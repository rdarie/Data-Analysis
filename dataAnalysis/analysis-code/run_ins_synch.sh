#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1,2,3,4,5

# Specify a job name:
#SBATCH -J ins_synch

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-ins_synch.stdout
#SBATCH -e ../batch_logs/%j-%a-ins_synch.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/synchronizeINStoNSP.py' --trialIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
