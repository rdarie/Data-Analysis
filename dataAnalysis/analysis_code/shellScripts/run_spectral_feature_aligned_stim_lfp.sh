#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=150G

# Specify a job name:
#SBATCH -J spectral_calc_stim_lfp_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-spectral_calc_stim_lfp_28.out
#SBATCH -e ../../batch_logs/%j-%a-spectral_calc_stim_lfp_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/calc_aligned_stim_preamble.sh

python -u ./calcWaveletFeatures.py --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $TIMEWINDOWOPTS $LAZINESS
python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

# python -u ./calcWaveletFeatures.py --inputBlockSuffix="lfp" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $TIMEWINDOWOPTS $LAZINESS
# python -u ./makeViewableBlockFromTriggered.py --plotting --inputBlockSuffix="lfp_spectral" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
##