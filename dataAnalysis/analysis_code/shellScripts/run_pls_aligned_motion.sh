#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pca_calc_stim_lapl

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_calc_stim_lapl.out
#SBATCH -e ../../batch_logs/%j-%a-pca_calc_stim_lapl.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh

# python -u './calcPartialLeastSquares.py' --lhsBlockSuffix="kcsd" --unitQueryLhs="lfp" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngle" --estimatorName='pls' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './calcPartialLeastSquares.py' --blockIdx=2 --processAll --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp" --rhsBlockSuffix="rig" --unitQueryRhs="pedalPosition" --estimatorName='pls' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER --plotting