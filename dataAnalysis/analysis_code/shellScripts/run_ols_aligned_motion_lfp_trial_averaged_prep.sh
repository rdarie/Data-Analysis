#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=127G

# Specify a job name:
#SBATCH -J ols_motion_lfp_prep_ta_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_lfp_prep_ta_28.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_lfp_prep_ta_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
ITERATOR="rb"
WINDOWTERM="XL"
################################################################################################################
## time domain, trial-averaged
SUFFIX=""
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
#
# ESTIMATOR="enr_pca_ta${SUFFIX}"
# python -u './prepSignalsAsRegressorV2.py' --transformerNameRhs='pca_ta' --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
# ESTIMATOR="enr_fa_ta${SUFFIX}"
# python -u './prepSignalsAsRegressorV2.py' --transformerNameRhs='fa_ta' --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
################################################################################################################
## spectral, trial-averaged
SUFFIX="_spectral"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
#
ESTIMATOR="enr_pca_ta${SUFFIX}"
python -u './prepSignalsAsRegressorV2.py' --transformerNameRhs='pca_ta' --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
ESTIMATOR="enr_fa_ta${SUFFIX}"
python -u './prepSignalsAsRegressorV2.py' --transformerNameRhs='fa_ta' --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
