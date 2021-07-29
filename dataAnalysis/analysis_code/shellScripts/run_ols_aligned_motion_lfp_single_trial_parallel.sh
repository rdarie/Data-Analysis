#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=4
#SBATCH --ntasks-per-core=4
#SBATCH --mem-per-cpu=8G

# Specify a job name:
#SBATCH -J ols_motion_lfp_pa_st_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_lfp_pa_st_28.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_lfp_pa_st_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=0-48
source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=2 --processAll"
ITERATOR="rb"
WINDOWTERM="XL"
SUFFIX=""
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
DIMRED="fa_ta"
ESTIMATOR="enr_${DIMRED}${SUFFIX}"
python -u './calcGridSearchRegressionWithPipelinesV2.py' --transformerNameRhs="${DIMRED}" --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
#   DIMRED="pca_ta"
#   ESTIMATOR="enr_${DIMRED}${SUFFIX}"
#   python -u './calcGridSearchRegressionWithPipelinesV2.py' --transformerNameRhs="${DIMRED}" --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1

#
#   SUFFIX="_spectral"
#   RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
#   DIMRED="pca_ta"
#   ESTIMATOR="enr_${DIMRED}${SUFFIX}"
#   python -u './calcGridSearchRegressionWithPipelinesV2.py' --transformerNameRhs="${DIMRED}" --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
#   DIMRED="fa_ta"
#   ESTIMATOR="enr_${DIMRED}${SUFFIX}"
#   python -u './calcGridSearchRegressionWithPipelinesV2.py' --transformerNameRhs="${DIMRED}" --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1