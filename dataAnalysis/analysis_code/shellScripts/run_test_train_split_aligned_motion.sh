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
#SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh

python -u './calcTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --needsRollingWindow --calcTimeROI --eventName='motion' --eventBlockSuffix='epochs' --ROIWinStart=100 --ROIWinStop=100 --timeROIAlignQuery='stopping' --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --selectionName='lfp_CAR' --needsRollingWindow --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --selectionName='lfp_CAR_spectral' --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd" --unitQuery="lfp" --selectionName='csd' --needsRollingWindow --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --selectionName='csd_spectral' --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="jointAngle" --selectionName='jointAngle' --needsRollingWindow --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
python -u './applyTestTrainSplit.py' --inputBlockSuffix="rig" --unitQuery="pedalPosition" --selectionName='pedalPosition' --needsRollingWindow --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
#
# python -u './calcTestTrainSplit.py' --blockIdx=2 --processAll --loadFromFrames --inputBlockSuffix="rig" --unitQuery="jointAngle" --selectionName='jointAngle' --verbose --iteratorSuffix='a' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
