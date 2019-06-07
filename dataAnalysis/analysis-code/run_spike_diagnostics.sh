#!/bin/bash
# 05: Assemble the spike nix file
# Request 24 hours of runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --array=1,2

# Specify a job name:
#SBATCH -J sorting_diagnostics

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-sorting-diagnostics.stdout
#SBATCH -e ../batch_logs/%j-%a-sorting-diagnostics.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID --makeCoarseNeoBlock
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/generateSpikeReport.py' --trialIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_coarse
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/generateSpikeReport.py' --trialIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final
