#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_assembler_covariances_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_assembler_covariances_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_assembler_covariances_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
# source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=2 --processAll"

ITERATOR="--iteratorSuffix=ca"
ALIGNQUERYTERM="outbound"
CONTROLSTATUS="--controlSet"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $LAZINESS $TIMEWINDOWOPTS
#
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR' --selectionName='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --selectionName='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

ITERATOR="--iteratorSuffix=cb"
ALIGNQUERYTERM="startingNoStim"
CONTROLSTATUS=""
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR' --selectionName='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --selectionName='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

ITERATOR="--iteratorSuffix=cc"
ALIGNQUERYTERM="startingE5"
CONTROLSTATUS=""
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR' --selectionName='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --selectionName='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
