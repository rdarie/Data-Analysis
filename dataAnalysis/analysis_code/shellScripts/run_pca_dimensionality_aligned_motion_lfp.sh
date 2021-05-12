#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pca_dimen_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_dimen_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-pca_dimen_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh
WINDOW="--window=L"
ALIGNQUERYTERM="startingNoStim"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="pedalState"
ITERATOR="c"

python -u './calcSignalDimensionality.py' --estimatorName="pca" --datasetName="lfp_CAR_spectral_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures
python -u './calcSignalDimensionality.py' --estimatorName="pca" --datasetName="lfp_CAR_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures