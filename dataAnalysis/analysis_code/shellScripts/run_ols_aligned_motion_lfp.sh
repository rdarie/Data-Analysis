#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=16
#SBATCH --mem-per-cpu=32G

# Specify a job name:
#SBATCH -J ols_motion_lfp_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_lfp_20.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_lfp_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="startingNoStim"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
ITERATOR="d"
WINDOWTERM="XL"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR_spectral"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig_regressor"

#
# python -u './calcGridSearchSingleTarget.py' --estimatorName='enr' --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#
# python -u './prepSignalsAsRegressor.py' --estimatorName='regressor' --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName='rig' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting --debugging

python -u './calcGridSearchRegressionWithPipelines.py' --transformerNameRhs='pca' --debugging --estimatorName='enr_spectral' --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './calcGridSearchRegressionWithPipelines.py' --transformerNameRhs='pca_ta' --debugging --estimatorName='enr_ta_spectral' --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#w
# python './processOrdinaryLeastSquares.py' --estimatorName='enr_spectral' --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
# sspython './processOrdinaryLeastSquares.py' --estimatorName='enr_ta_spectral' --datasetName=Block_${WINDOWTERM}_df_${ITERATOR} --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
