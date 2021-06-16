#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J test_train_split_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split_28.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

source shellScripts/calc_aligned_motion_preamble.sh

#  ITERATOR="--iteratorSuffix=a"
#  ALIGNQUERYTERM="outbound"
#  CONTROLSTATUS="--controlSet"
###
#  ITERATOR="--iteratorSuffix=b"
#  ALIGNQUERYTERM="startingNoStim"
#  CONTROLSTATUS=""
##
ITERATOR="--iteratorSuffix=c"
ALIGNQUERYTERM="starting"
CONTROLSTATUS=""
#
# used d in the aligned to stim category############
#
# ITERATOR="--iteratorSuffix=e"
# ROIOPTS="--calcTimeROI --ROIWinStart=-100 --ROIWinStop=100 --timeROIAlignQuery=stopping"
# ALIGNQUERYTERM="startingNoStim"


ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
###
python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS
###
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
###
# python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="csd" --unitQuery="lfp" --selectionName='csd' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --selectionName='csd_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="limbState" --selectionName='limbState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

# next, go to run_test_train_split_assembler_xxx.sh