#!/bin/bash

# Request runtime:
#SBATCH --time=48:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=16G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J dimen_red_st_27

# Specify an output file
#SBATCH -o ../../batch_logs/dimen_red_st_27.out
#SBATCH -e ../../batch_logs/dimen_red_st_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
##################################################
ITERATOR="ra"
##################################################

# targets=(lfp_CAR lfp_CAR_spectral_scaled)
# estimators=(fa pca)
#
targets=(lfp_CAR)
estimators=(fa)
for TARGET in "${targets[@]}"
do
  for ESTIMATOR in "${estimators[@]}"
  do
    python -u './calcGridSearchSignalDimensionalityV2.py' --debugging --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
    python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  done
done