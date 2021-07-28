#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_assembler_regression_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_assembler_regression_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_assembler_regression_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
# source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
# ITERATOR="ra"
# ALIGNQUERYTERM="starting"
# ITERATOR="rb"
# ALIGNQUERYTERM="starting"
ITERATOR="rc"
# ALIGNQUERYTERM="starting"

# ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
#
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
# --preScale
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR' --selectionName='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --selectionName='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#