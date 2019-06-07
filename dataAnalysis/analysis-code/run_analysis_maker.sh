#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --array=1,2

# Specify a job name:
#SBATCH -J analysis_maker

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-anmaker.stdout
#SBATCH -e ../batch_logs/%j-%a-anmaker.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/makeTrialAnalysisNix.py' --trialIdx=$SLURM_ARRAY_TASK_ID
