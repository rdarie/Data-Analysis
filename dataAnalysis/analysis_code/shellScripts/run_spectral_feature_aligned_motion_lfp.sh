#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J spectral_calc_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spectral_calc_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-spectral_calc_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/calc_aligned_motion_preamble.sh
python -u ./calcWaveletFeatures.py --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $TIMEWINDOWOPTS $LAZINESS
python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

