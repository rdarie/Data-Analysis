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
#SBATCH -J ols_motion_lfp_post_predictions_ta_27

# Specify an output file
#SBATCH -o ../../batch_logs/ols_motion_lfp_post_predictions_ta_27-%a.out
#SBATCH -e ../../batch_logs/ols_motion_lfp_post_predictions_ta_27-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=0-56

source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=2 --processAll"
ITERATOR="ra"
WINDOWTERM="XL"
#
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"

#  --forceReprocess
ESTIMATOR="enr_fa_ta"
python -u './processOrdinaryLeastSquaresPredictions_parallelV2.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
