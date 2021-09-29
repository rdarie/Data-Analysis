#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J tts_norm_stim_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/test_train_split_normalization_stim_202101_20-%a.out
#SBATCH -e ../../batch_logs/test_train_split_normalization_stim_202101_20-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=2
source shellScripts/run_exp_preamble_202101_20.sh
source shellScripts/calc_aligned_stim_preamble.sh
###

ITERATOR="--iteratorSuffix=na"
ALIGNQUERYTERM="stimOnHighRate"
CONTROLSTATUS=""
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ROIOPTS $ITERATOR --eventName='stim' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
##
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace" --unitQuery="laplace" --selectionName='laplace' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS

# next, go to run_test_train_split_assembler_xxx.sh