#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pls_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pls_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-pls_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

python -u './calcPartialLeastSquares.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngle" --estimatorName='pls_lfp_CAR_ja' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './calcPartialLeastSquares.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngle" --estimatorName='pls_lfp_CAR_spectral_ja' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
