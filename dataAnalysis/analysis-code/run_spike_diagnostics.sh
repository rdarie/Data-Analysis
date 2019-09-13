#!/bin/bash
# 05: Assemble the spike nix file
# Request 24 hours of runtime:
#SBATCH --time=4:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J sorting_diagnostics

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-sorting-diagnostics.stdout
#SBATCH -e ../batch_logs/%j-%a-sorting-diagnostics.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

EXP="exp201901211000"
# EXP="exp201901201200"
# EXP="exp201901221000"
# EXP="exp201901070700"
# EXP="exp201901271000"
# EXP="exp201901231000"

#SLURM_ARRAY_TASK_ID="2"
#python3 './tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID --makeCoarseNeoBlock --exp=$EXP
#python3 './plotSpikeReport.py' --trialIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_coarse --exp=$EXP
python3 './tridesclousCCV.py' --trialIdx=$SLURM_ARRAY_TASK_ID --makeStrictNeoBlock --exp=$EXP
python3 './plotSpikeReport.py' --trialIdx=$SLURM_ARRAY_TASK_ID --nameSuffix=_final --exp=$EXP
