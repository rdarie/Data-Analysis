#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=96G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J s08_pls_tf_sta_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/regression/job_arrays/s08_pls_tf_sta_202101_20-%a.out
#SBATCH -e ../../batch_logs/regression/job_arrays/s08_pls_tf_sta_202101_20-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=0-47

# exps=(201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
#    SLURM_ARRAY_TASK_ID=12
exps=(202101_20)
for A in "${exps[@]}"
do
  echo "step 08 transfer function calc, on $A"
  source shellScripts/run_exp_preamble_$A.sh
  source shellScripts/calc_aligned_motion_preamble.sh
  #
  ALIGNQUERYTERM="starting"
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  SUFFIX="_scaled"
  ITERATOR="ra"
  WINDOWTERM="XL"
  #
  RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=laplace${SUFFIX}"
  LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
  #  --forceReprocess
  DIMRED="select"
  ESTIMATOR="ols_${DIMRED}${SUFFIX}"
  python -u './processLeastSquaresTransferFunction.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done