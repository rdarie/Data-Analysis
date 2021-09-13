#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J ols_motion_lfp_prep_ta_td_fa_28

# Specify an output file
#SBATCH -o ../../batch_logs/ols_motion_lfp_prep_ta_td_fa_28.out
#SBATCH -e ../../batch_logs/ols_motion_lfp_prep_ta_td_fa_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_temp.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

ITERATOR="ra"
WINDOWTERM="XL"
################################################################################################################
#
## time domain, trial-averaged
SUFFIX=""
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
ESTIMATOR="enr_fa_ta${SUFFIX}"
python -u './prepSignalsAsRegressorV3.py' --transformerNameRhs='fa_ta' --maxNumFeatures=8 --debugging --estimatorName=$ESTIMATOR --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1