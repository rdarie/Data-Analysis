#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J apply_mahal_dist_stim_lfp_25

# Specify an output file
#SBATCH -o ../../batch_logs/apply_mahal_dist_stim_lfp_25-%a.out
#SBATCH -e ../../batch_logs/apply_mahal_dist_stim_lfp_25-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1

#   SLURM_ARRAY_TASK_ID=1

source ./shellScripts/run_exp_preamble_temp.sh
source ./shellScripts/calc_aligned_stim_preamble.sh

####################
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
ITERATOR="ca"
#
ALIGNQUERYTERM="stimOn"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
#
targets=(lfp_CAR lfp_CAR_spectral_scaled)
estimators=(mahal_ledoit)
#
for TARGET in "${targets[@]}"
do
  for ESTIMATOR in "${estimators[@]}"
  do
    python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS
    python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="mahal" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  done
done