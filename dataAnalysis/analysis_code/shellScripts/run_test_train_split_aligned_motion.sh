#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J test_train_split

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=3
source shellScripts/calc_aligned_motion_preamble.sh

# ITERATOR="--iteratorSuffix=a"
# ROIOPTS="--calcTimeROI --ROIWinStart=0 --ROIWinStop=0 --timeROIAlignQuery=stopping"
# ITERATOR="--iteratorSuffix=b"
# ROIOPTS="--calcTimeROI --ROIWinStart=-50 --ROIWinStop=250 --timeROIAlignQuery=starting"
ITERATOR="--iteratorSuffix=c"
ROIOPTS="--calcTimeROI --ROIWinStart=-50 --ROIWinStop=0 --timeROIAlignQuery=stopping"

python -u './calcTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ROIOPTS $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS

python -u './applyTestTrainSplit.py' --resetHDF --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --verbose $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd" --unitQuery="lfp" --selectionName='csd' --verbose $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --selectionName='csd_spectral' --verbose $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="limbState" --selectionName='limbState' --verbose $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' $ITERATOR --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
#
# next, go to run_test_train_split_assembler_xxx.sh