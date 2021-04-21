#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J ols_motion_csd

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_csd.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_csd.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# python -u './calcOrdinaryLeastSquares.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="csd" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngle" --estimatorName='ols' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures
python -u './calcOrdinaryLeastSquares.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="csd_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngle" --estimatorName='ols' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures
python -u './processOrdinaryLeastSquares.py' --fullEstimatorName='ols_csd_spectral_to_jointAngle_a_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures
python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName='ols_csd_spectral_to_jointAngle_a_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --showFigures