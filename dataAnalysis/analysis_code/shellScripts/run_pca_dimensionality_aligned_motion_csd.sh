#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pca_dimen_motion_csd

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_dimen_motion_csd.out
#SBATCH -e ../../batch_logs/%j-%a-pca_dimen_motion_csd.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
python -u './calcSignalDimensionality.py' --loadFromFrames --inputBlockSuffix="csd" --unitQuery="csd" --estimatorName="pca_csd" --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './calcSignalDimensionality.py' --loadFromFrames --inputBlockSuffix="csd_spectral" --unitQuery="csd_spectral" --estimatorName="pca_csd_spectral" --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
