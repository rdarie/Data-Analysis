#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_split

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#                 SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="lfp_CAR_spectral_fa_mahal"

ITERATOR="f"
ALIGNQUERYTERM="starting"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix=$TARGET --unitQuery="mahal" --selectionName=$TARGET --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR_spectral_mahal' --selectionName=$TARGET --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#