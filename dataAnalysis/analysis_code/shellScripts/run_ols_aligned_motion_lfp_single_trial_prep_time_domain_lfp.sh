#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J ols_motion_lfp_prep_lfp_st_27

# Specify an output file
#SBATCH -o ../../batch_logs/ols_motion_lfp_prep_lfp_st_27.out
#SBATCH -e ../../batch_logs/ols_motion_lfp_prep_lfp_st_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

ITERATOR="ra"
WINDOWTERM="XL"
##
ESTIMATOR="select"
TARGET="lfp_CAR_scaled"
python -u './calcSignalColumnSelector.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#
################################################################################################################
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR_scaled"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
#
ESTIMATOR="enr_select"
python -u './prepSignalsAsRegressorV2.py' --transformerNameRhs='select' --maxNumFeatures=8 --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1