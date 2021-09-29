#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J tts_assembler_normalization_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/test_train_assembler_normalization_202101_20.out
#SBATCH -e ../../batch_logs/test_train_assembler_normalization_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
# source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
ITERATOR="na"
#
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="laplace_spectral" --unitQuery="lfp" --selectionName='laplace_spectral' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
# --preScale
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace' --selectionName='laplace' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_spectral' --selectionName='laplace_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#