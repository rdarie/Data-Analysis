#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=7
#SBATCH --mem-per-cpu=5G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J s06a_ols_rc_202101_22

# Specify an output file
#SBATCH -o ../../batch_logs/regression/job_arrays/s06a_ols_rc_202101_22-%a.out
#SBATCH -e ../../batch_logs/regression/job_arrays/s06a_ols_rc_202101_22-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1
# Request custom resources
#SBATCH --array=16-27,91-96

# exps=(201901_25 201901_27 201901_25 201902 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_22)
for A in "${exps[@]}"
do
  echo "step 06 pls regression and predictions, on $A"
  source shellScripts/run_exp_preamble_$A.sh
  source shellScripts/calc_aligned_motion_preamble.sh
  ALIGNQUERYTERM="starting"
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="rc"
  WINDOWTERM="XL"
  SUFFIX="_scaled"
  RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=laplace${SUFFIX}"
  LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
  DIMRED="select"
  # ols_select_scaled
  ESTIMATOR="ols_${DIMRED}${SUFFIX}"
  python -u './calcGridSearchRegressionWithPipelinesV5.py' --transformerNameRhs="${DIMRED}" --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=5
  python -u './calcOrdinaryLeastSquaresPredictionsLite.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # python -u './calcOrdinaryLeastSquaresPredictionsFull.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done