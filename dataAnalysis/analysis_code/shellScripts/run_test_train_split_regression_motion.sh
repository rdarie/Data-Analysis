#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J test_train_split_regression_motion_27

# Specify an output file
#SBATCH -o ../../batch_logs/regression/test_train_split_regression_motion_27-%a.out
#SBATCH -e ../../batch_logs/regression/test_train_split_regression_motion_27-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3
#SBATCH --export=CCV_HEADLESS=1

#     SLURM_ARRAY_TASK_ID=3
source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_motion_preamble.sh

ITERATOR="--iteratorSuffix=ra"
ALIGNQUERYTERM="starting"
CONTROLSTATUS=""


ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
###
python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
###
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="lfp_CAR_scaled" --unitQuery="lfp" --selectionName='lfp_CAR_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
#  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_CAR_spectral_scaled" --unitQuery="lfp" --selectionName='lfp_CAR_spectral_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

# next, go to run_test_train_split_assembler_xxx.sh