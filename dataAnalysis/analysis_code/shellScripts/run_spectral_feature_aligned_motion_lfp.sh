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
SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh
# python -u './calcSignalDimensionality.py' --loadFromFrames --inputBlockSuffix="lfp" --unitQuery="lfp" --estimatorName="pca_lfp" --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './calcWaveletFeatures.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $TIMEWINDOWOPTS $LAZINESS --verbose