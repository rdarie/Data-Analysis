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
TARGET="pedalState"
ITERATOR="b"

# iterator "b", -50 to 250 msec after start

# python -u './calcOrdinaryLeastSquares.py' --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --estimatorName='ols' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $VERBOSITY --plotting --showFigures
# python -u './processOrdinaryLeastSquares.py' --fullEstimatorName="ols_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
# python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName="ols_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting

# python -u './assembleRegressionTerms.py' --debugging --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

#
# python -u './calcGridSearchSingleTarget.py' --debugging --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --estimatorName='enr' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './processOrdinaryLeastSquares.py' --debugging --estimatorName="enr" --datasetName="lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --estimatorName="enr" --datasetName="lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting

# python -u './assembleRegressionTerms.py' --debugging --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#
# python -u './calcGridSearchSingleTarget.py' --debugging --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --estimatorName='enr' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './processOrdinaryLeastSquares.py' --debugging --estimatorName="enr" --datasetName="lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --estimatorName="enr" --datasetName="lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_${ALIGNQUERYTERM}" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
