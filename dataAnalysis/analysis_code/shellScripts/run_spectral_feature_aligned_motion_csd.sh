#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J spectral_calc_motion_csd

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spectral_calc_motion_csd.out
#SBATCH -e ../../batch_logs/%j-%a-spectral_calc_motion_csd.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
python -u './calcWaveletFeatures.py' --inputBlockSuffix="csd" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $TIMEWINDOWOPTS $LAZINESS --verbose
