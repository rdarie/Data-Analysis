#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=8G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J ols_motion_lfp_st_27

# Specify an output file
#SBATCH -o ../../batch_logs/ols_motion_lfp_st_27.out
#SBATCH -e ../../batch_logs/ols_motion_lfp_st_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=0-56
source shellScripts/run_exp_preamble_temp.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=2 --processAll"
ITERATOR="ra"
WINDOWTERM="XL"
SUFFIX=""
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
DIMRED="fa"
ESTIMATOR="enr_${DIMRED}${SUFFIX}"
python -u './calcGridSearchRegressionWithPipelinesV2.py' --transformerNameRhs="${DIMRED}" --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
