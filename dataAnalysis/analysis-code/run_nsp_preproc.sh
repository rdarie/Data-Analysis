#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --array=2,3,4

# Specify a job name:
#SBATCH -J nsp_preproc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-nsp_preproc.stdout
#SBATCH -e ../batch_logs/%j-%a-nsp_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# EXP="exp201901221000"
# EXP="exp201901231000"

python3 './preprocNS5.py' --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --makeTruncated
