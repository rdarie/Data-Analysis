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
#               SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="pedalState"

ITERATOR="a"
ALIGNQUERYTERM="outbound"
#
# ITERATOR="b"
# ALIGNQUERYTERM="startingNoStim"
#
# ITERATOR="c"
# ALIGNQUERYTERM="startingE5"
#
# ITERATOR="e"
# ALIGNQUERYTERM="startingNoStim"
#####
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
#
# python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#
# python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR' --selectionName='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --selectionName='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#
python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="lfp_CAR"          $WINDOW $ANALYSISFOLDER --alignFolderName=motion $LAZINESS
python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="lfp_CAR"          $WINDOW $ANALYSISFOLDER --alignFolderName=stim $LAZINESS
python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="lfp_CAR_spectral" $WINDOW $ANALYSISFOLDER --alignFolderName=motion $LAZINESS
python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="lfp_CAR_spectral" $WINDOW $ANALYSISFOLDER --alignFolderName=stim $LAZINESS
