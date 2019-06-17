#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --array=1,2,3,4,5

# Specify a job name:
#SBATCH -J analysis_calc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-analysis_calc.stdout
#SBATCH -e ../batch_logs/%j-%a-analysis_calc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901201200_alt"
# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcTrialAnalysisNix.py' --trialIdx=$SLURM_ARRAY_TASK_ID  --exp=$EXP
