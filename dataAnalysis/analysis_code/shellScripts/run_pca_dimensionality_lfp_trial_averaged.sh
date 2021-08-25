#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J dimen_red_ta_25

# Specify an output file
#SBATCH -o ../../batch_logs/dimen_red_ta_25.out
#SBATCH -e ../../batch_logs/dimen_red_ta_25.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_temp.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
##################################################
ITERATOR="ra"
##################################################

# targets=(lfp_CAR lfp_CAR_spectral lfp_CAR_spectral_scaled)
# estimators=(fa_ta pca_ta)
#
targets=(lfp_CAR)
estimators=(fa_ta)
for TARGET in "${targets[@]}"
do
  for ESTIMATOR in "${estimators[@]}"
  do
    python -u './calcGridSearchSignalDimensionalityV2.py' --debugging --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
    python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  done
done