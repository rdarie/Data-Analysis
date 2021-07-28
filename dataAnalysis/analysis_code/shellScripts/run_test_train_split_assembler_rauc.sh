#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_split_assembler_rauc_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_splitt_assembler_rauc_27.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_splitt_assembler_rauc_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#                 SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
ITERATOR="ma"
ESTIMATOR='mahal_ledoit'

python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
# --preScale
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix="lfp_CAR_${ESTIMATOR}" --selectionName="lfp_CAR_${ESTIMATOR}" --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix="lfp_CAR_spectral_${ESTIMATOR}" --selectionName="lfp_CAR_spectral_${ESTIMATOR}" --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
