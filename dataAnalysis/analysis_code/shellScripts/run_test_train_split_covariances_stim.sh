#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J test_train_split_covariances_stim_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split_covariances_stim_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split_covariances_stim_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=2
source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_stim_preamble.sh
###

BLOCKSELECTOR="--blockIdx=1"
ITERATOR="--iteratorSuffix=cc"
ALIGNQUERYTERM="stimOnE5"
CONTROLSTATUS=""
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ROIOPTS $ITERATOR --eventName='stim' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
##
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
#

source shellScripts/calc_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=2"

ITERATOR="--iteratorSuffix=cc"
ALIGNQUERYTERM="startingE5"
CONTROLSTATUS=""
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
###
python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
###
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS



# next, go to run_test_train_split_assembler_xxx.sh