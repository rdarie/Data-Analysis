#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --array=1,2

# Specify a job name:
#SBATCH -J ins_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-ins_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-ins_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/preprocINS.py' --trialIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP