#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=150G

# Specify a job name:
#SBATCH -J spectral_calc_motion_lfp_201901_26

# Specify an output file
#SBATCH -o ../../batch_logs/spectral_calc_motion_lfp_201901_26-%a.out
#SBATCH -e ../../batch_logs/spectral_calc_motion_lfp_201901_26-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1-3

#  SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_exp_preamble_26.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
#
python -u ./calcWaveletFeatures.py --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u ./makeViewableBlockFromTriggered.py --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
##
# python -u ./calcWaveletFeatures.py  --plotting --inputBlockSuffix="lfp" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u ./makeViewableBlockFromTriggered.py --inputBlockSuffix="lfp_spectral" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

