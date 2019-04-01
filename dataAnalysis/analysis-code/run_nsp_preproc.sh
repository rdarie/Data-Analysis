#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --array=1,2,3,4

# Specify a job name:
#SBATCH -J nsp_preproc

# Specify an output file
#SBATCH -o ../batch_logs/preprocnsp-o-%j-%a.out
#SBATCH -e ../batch_logs/preprocnsp-e-%j-%a.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/preprocNS5.py' --trialIdx=$SLURM_ARRAY_TASK_ID
