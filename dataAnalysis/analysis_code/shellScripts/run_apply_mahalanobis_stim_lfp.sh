#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J apply_mahalanobis_stim_lfp_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-apply_mahalanobis_stim_lfp_28.out
#SBATCH -e ../../batch_logs/%j-%a-apply_mahalanobis_stim_lfp_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1

#  SLURM_ARRAY_TASK_ID=1

source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_stim_preamble.sh

####################
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
ITERATOR="ca"

ALIGNQUERYTERM="stimOn"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

TARGET="lfp_CAR_spectral"
ESTIMATOR="mahal"
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --datasetExp='202101281100-Rupert' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="mahal" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
