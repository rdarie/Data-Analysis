#!/bin/bash

# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=200G

# Specify a job name:
#SBATCH -J pls_post_impulses_sta_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/regression/pls_post_impulses_sta_202101_20.out
#SBATCH -e ../../batch_logs/regression/pls_post_impulses_sta_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source shellScripts/run_exp_preamble_202101_20.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
###
ITERATOR="ra"
###
WINDOWTERM="XL"
#
SUFFIX="_scaled"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR${SUFFIX}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"

#  --forceReprocess
DIMRED="select"
ESTIMATOR="pls_${DIMRED}${SUFFIX}"
python -u './processPartialLeastSquaresImpulses.py' --estimatorName=$ESTIMATOR --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
