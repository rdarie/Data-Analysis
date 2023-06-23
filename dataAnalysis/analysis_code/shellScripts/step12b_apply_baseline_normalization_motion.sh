#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s12b_apply_baseline_motion_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/s12b_apply_baseline_motion_202101_27-%a.out
#SBATCH -e ../../batch_logs/s12b_apply_baseline_motion_202101_27-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2-3

# 201902 201902_04 201902_05
# exps=(201901_25 201901_26 201901_27)
# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 04 apply normalization, motion,  get data for covariance, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  
  ####################
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
  #
  ITERATOR="cd"
  ALIGNQUERYTERM="starting"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  COMMONOPTS="--datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $VERBOSITY $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
  TARGET="laplace"
  ESTIMATOR="baseline"
  #
  python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
  #
  TARGET="laplace_spectral"
  ESTIMATOR="baseline"
  #
  python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --selectionName=$TARGET $COMMONOPTS $TRAINDATASET
done