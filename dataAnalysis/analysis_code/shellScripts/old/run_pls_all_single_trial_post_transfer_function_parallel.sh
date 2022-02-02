#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J pls_tf_sta_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/regression/job_arrays/pls_tf_sta_202101_20-%a.out
#SBATCH -e ../../batch_logs/regression/job_arrays/pls_tf_sta_202101_20-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=0-2

source shellScripts/run_exp_preamble_202101_20.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=2 --processAll"
SUFFIX="_scaled"
ITERATOR="ra"
###
WINDOWTERM="XL"
#
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=laplace${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"

#  --forceReprocess
DIMRED="select"
ESTIMATOR="pls_${DIMRED}${SUFFIX}"
python -u './processOrdinaryLeastSquaresTransferFunction_parallel.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
