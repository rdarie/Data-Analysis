#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J ols_motion_lfp_prep_lfp_ta_27

# Specify an output file
#SBATCH -o ../../batch_logs/ols_motion_lfp_prep_lfp_ta_27.out
#SBATCH -e ../../batch_logs/ols_motion_lfp_prep_lfp_ta_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

ITERATOR="rb"
WINDOWTERM="XL"
##
ESTIMATOR="select_ta"
TARGET="lfp_CAR_scaled"
python -u './calcSignalColumnSelector.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#
################################################################################################################
## time domain, trial-averaged

RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR_scaled"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
#
ESTIMATOR="enr2_select_ta"
python -u './prepSignalsAsRegressorV3.py' --transformerNameRhs='select_ta' --maxNumFeatures=16 --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1