#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=32G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J tts_assembler_regression_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/regression/test_train_assembler_regression_202101_20.out
#SBATCH -e ../../batch_logs/regression/test_train_assembler_regression_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
# source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

ITERATOR="ra"
# # 
# # python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
# # # --preScale
# # python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# # ##  python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# # ##  python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# # python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# # #
# # OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
# # python -u './plotFinalSynchConfirmation.py' --plotSuffix="final_synch_confirmation" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --winStart="-300" --winStop="500" --verbose=1 --selectionName="rig" --selectionName2="laplace_scaled" $OPTS

# # 
# # ESTIMATOR="select"
# # TARGET="laplace_scaled"
# # python -u './calcSignalColumnSelector.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#
################################################################################################################
# # RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=laplace_scaled"
# # LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
# # #
# # python -u './prepSignalsAsRegressorV3.py' --transformerNameRhs='select' --debugging --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
