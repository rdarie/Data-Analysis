#!/bin/bash

# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=8
#SBATCH --mem=16G
#SBATCH --array=1,2,3,4,5

# Specify a job name:
#SBATCH -J spike_sorting

# Specify an output file
#SBATCH -o ../batch_logs/sorting-o-%j-%a.out
#SBATCH -e ../batch_logs/sorting-e-%j-%a.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID --makeNeoBlock
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/generateSpikeReport.py' --trialIdx=$SLURM_ARRAY_TASK_ID

