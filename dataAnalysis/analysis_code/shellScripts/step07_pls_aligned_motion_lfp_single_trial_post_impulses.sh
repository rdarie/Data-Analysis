#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=200G

# Specify a job name:
#SBATCH -J s07_pls_post_impulses_sta_201901_27

# Specify an output file
#SBATCH -o ../../batch_logs/regression/s07_pls_post_impulses_sta_201901_27.out
#SBATCH -e ../../batch_logs/regression/s07_pls_post_impulses_sta_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

#  SLURM_ARRAY_TASK_ID=2
# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 07 impulse responses, scores, predictions, on $A"
  source shellScripts/run_exp_preamble_$A.sh
  source shellScripts/calc_aligned_motion_preamble.sh
  
  ALIGNQUERYTERM="starting"
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ###
  ITERATOR="ra"
  WINDOWTERM="XL"
  SUFFIX="_scaled"
  RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
  LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
  #  --forceReprocess
  DIMRED="select"
  ESTIMATOR="ols_${DIMRED}${SUFFIX}"
  python -u './processPartialLeastSquaresImpulses.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # 
  python -u './processPartialLeastSquaresScoresVP1.py' --memoryEfficientLoad --forceReprocess --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  python -u './processPartialLeastSquaresPredictionsLite.py' --memoryEfficientLoad --forceReprocess --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  python -u './plotPartialLeastSquaresPredictions.py' --memoryEfficientLoad --forceReprocess --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  #
  # python -u './processOrdinaryLeastSquaresPaperPlots.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done