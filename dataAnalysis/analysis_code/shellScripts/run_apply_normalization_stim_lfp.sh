#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J apply_spectral_norm_stim_lfp_26

# Specify an output file
#SBATCH -o ../../batch_logs/apply_spectral_norm_stim_lfp_26-%a.out
#SBATCH -e ../../batch_logs/apply_spectral_norm_stim_lfp_26-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=4

#   SLURM_ARRAY_TASK_ID=1

source ./shellScripts/run_exp_preamble_26.sh
source ./shellScripts/calc_aligned_stim_preamble.sh

####################
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
ITERATOR="na"
#
ALIGNQUERYTERM="stimOn"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
COMMONOPTS="--datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $VERBOSITY $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS"
TARGET=lfp_CAR
ESTIMATOR=scaled
TRAINDATASET=""
TRAINDATASET="--datasetExp=201901271000-Murdoc"
#
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
# python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

TARGET=lfp_CAR_spectral
ESTIMATOR=scaled
#
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
# python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
