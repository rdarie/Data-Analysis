#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J apply_pca_dimen_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-apply_pca_dimen_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-apply_pca_dimen_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/calc_aligned_motion_preamble.sh


BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
TARGET="lfp_CAR_spectral"
ESTIMATOR="fa"
ITERATOR="a"
WINDOW="XL"

# python -u './applyEstimatorToTriggered.py' --matchDownsampling --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --datasetName="${TARGET}_a_XL_outbound" --datasetExp='202101281100-Rupert' --unitQuery="${TARGET}" --exp=$EXP --window=$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="factor" $VERBOSITY --exp=$EXP --window=$WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

TARGET="lfp_CAR_spectral_fa"
ESTIMATOR="mahal"
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --datasetName="${TARGET}_a_XL_outbound" --datasetExp='202101281100-Rupert' --exp=$EXP --window=$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="mahal" $VERBOSITY --exp=$EXP --window=$WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
