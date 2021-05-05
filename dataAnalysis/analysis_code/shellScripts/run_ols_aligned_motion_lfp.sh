#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=16
#SBATCH --mem-per-cpu=32G

# Specify a job name:
#SBATCH -J ols_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
#   SLURM_ARRAY_TASK_ID=2
source shellScripts/run_plotter_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="pedalState"

ITERATOR="b"

# iterator "b", -50 to 250 msec after start

python -u './calcOrdinaryLeastSquares.py' --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs=$TARGET --estimatorName='ols' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './processOrdinaryLeastSquares.py' --fullEstimatorName="ols_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_starting" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName="ols_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_starting" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting

python -u './calcGridSearchSingleTarget.py' --iteratorSuffix=$ITERATOR --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngularVelocity" --estimatorName='enr' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './processOrdinaryLeastSquares.py' --fullEstimatorName="enr_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_starting" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName="enr_lfp_CAR_spectral_to_${TARGET}_${ITERATOR}_L_starting" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
