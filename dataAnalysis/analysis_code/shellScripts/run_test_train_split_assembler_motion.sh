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

ALIGNQUERYTERM="starting"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
TARGET="pedalState"
ITERATOR="e"

python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS

python -u './assembleDataFrames.py' --debugging --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR' --unitQuery='lfp_CAR' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --debugging --iteratorSuffix=$ITERATOR --inputBlockSuffix='lfp_CAR_spectral' --unitQuery='lfp_CAR_spectral' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --debugging --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --unitQuery='pedalState' --loadFromFrames --exp=$EXP $WINDOW --alignQuery=$ALIGNQUERYTERM $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

#
# blocks=(lfp_CAR lfp_CAR_spectral rig)
# for B in "${blocks[@]}"
# do
#     echo "concatenating $B blocks"
#     python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER --alignQuery=$ALIGNQUERYTERM $LAZINESS
# done