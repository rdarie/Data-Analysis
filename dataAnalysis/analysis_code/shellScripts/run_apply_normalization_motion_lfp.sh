#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J apply_norm_motion_lfp_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/apply_norm_motion_lfp_202101_20-%a.out
#SBATCH -e ../../batch_logs/apply_norm_motion_lfp_202101_20-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2-3

source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

####################
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
ITERATOR="na"
#
ALIGNQUERYTERM="starting"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
COMMONOPTS="--datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $VERBOSITY $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
TARGET=laplace
ESTIMATOR=scaled
TRAINDATASET=""
# TRAINDATASET="--datasetExp=201901271000-Murdoc"
#
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
# python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

TARGET=laplace_spectral
ESTIMATOR=scaled
#
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
# python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
