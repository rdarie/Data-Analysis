#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J test_train_split_rauc_stim_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split_rauc_stim_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split_rauc_stim_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1

#     SLURM_ARRAY_TASK_ID=1
source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_stim_preamble.sh
#
ITERATOR="--iteratorSuffix=ma"
#
ALIGNQUERYTERM="stimOnHighRate"
CONTROLSTATUS=""
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
ESTIMATOR='mahal_ledoit'

python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ROIOPTS $ITERATOR --eventName='stim' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
###
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="lfp_CAR_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_CAR_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_CAR_spectral_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_CAR_spectral_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName="rig" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
