#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --array=2,3,4

# Specify a job name:
#SBATCH -J analysis_calc

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-analysis_calc.stdout
#SBATCH -e ../batch_logs/%j-%a-analysis_calc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901201200"
# EXP="exp201901211000"
EXP="exp201901271000"

python3 ./calcTrialAnalysisNix.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --chanQuery="all"
python3 ./calcMotionStimAlignTimes.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID --plotParamHistograms
# python3 ./calcFR.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID
# python3 ./calcFRsqrt.py --exp=$EXP --trialIdx=$SLURM_ARRAY_TASK_ID
