#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_split_b

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split_b.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split_b.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=3
source ./shellScripts/calc_aligned_motion_preamble.sh

python -u './calcTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --calcTimeROI --ROIWinStart=-50 --ROIWinStop=300 --eventName='motion' --eventBlockSuffix='epochs' --timeROIAlignQuery='starting' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS $VERBOSITY
#
# python -u './applyTestTrainSplit.py' --resetHDF --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
python -u './applyTestTrainSplit.py' --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd" --unitQuery="lfp" --selectionName='csd' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --selectionName='csd_spectral' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="limbState" --selectionName='limbState' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY
# python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --iteratorSuffix='b' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS $VERBOSITY

# next, go to run_test_train_split_assembler_xxx.sh