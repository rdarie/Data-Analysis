#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=4
#SBATCH --ntasks-per-core=4
#SBATCH --mem-per-cpu=72G

# Specify a job name:
#SBATCH -J dimen_red_ta_rc_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-dimen_red_ta_rc_27.out
#SBATCH -e ../../batch_logs/%j-%a-dimen_red_ta_rc_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
##################################################
# ITERATOR="ra"
# ITERATOR="rb"
ITERATOR="rc"
##################################################

#####

# targets=(lfp_CAR lfp_CAR_spectral)
# estimators=(fa_ta pca_ta)
#
targets=(lfp_CAR)
estimators=(fa_ta)
for TARGET in "${targets[@]}"
do
  # python -u './testSignalNormality.py' --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  for ESTIMATOR in "${estimators[@]}"
  do
    python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
    python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  done
done