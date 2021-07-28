#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_split_assembler_rauc_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_splitt_assembler_rauc_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_splitt_assembler_rauc_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#                 SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
ITERATOR="--iteratorSuffix=ma"
ALIGNQUERYTERM="starting"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

TARGET="lfp_CAR_mahal"
#
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix=$TARGET --unitQuery="mahal" --selectionName=$TARGET --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
#
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix=$TARGET --selectionName=$TARGET --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

TARGET="lfp_CAR_spectral_mahal"
#
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix=$TARGET --unitQuery="mahal" --selectionName=$TARGET --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
#
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix=$TARGET --selectionName=$TARGET --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
