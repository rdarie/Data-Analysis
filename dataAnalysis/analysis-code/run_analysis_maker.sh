#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=5

# Specify a job name:
#SBATCH -J analysis_maker

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-anmaker.stdout
#SBATCH -e ../batch_logs/%j-%a-anmaker.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/makeTrialAnalysisNix.py' --trialIdx=$SLURM_ARRAY_TASK_ID
