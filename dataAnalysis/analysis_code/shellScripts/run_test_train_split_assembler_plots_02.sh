#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J tts_assembler_plots_02_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/test_train_split_assembler_plots_02_202101_20.out
#SBATCH -e ../../batch_logs/test_train_split_assembler_plots_02_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
ITERATOR="pa"

python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
# --preScale
COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
python -u './assembleDataFrames.py' --inputBlockSuffix="laplace" --selectionName="laplace" $COMMONOPTS
python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral" --selectionName="laplace_spectral" $COMMONOPTS
# python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral_scaled" --selectionName="laplace_spectral_scaled" $COMMONOPTS
python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS

# ESTIMATOR='mahal_ledoit'
# python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_${ESTIMATOR}" --selectionName="laplace_${ESTIMATOR}" $COMMONOPTS
# python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral_scaled_${ESTIMATOR}" --selectionName="laplace_spectral_scaled_${ESTIMATOR}" $COMMONOPTS
