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
#SBATCH -J tts_assembler_covariances_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/test_train_assembler_covariances_202101_20.out
#SBATCH -e ../../batch_logs/test_train_assembler_covariances_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh
#
BLOCKSELECTOR="--blockIdx=2 --processAll"
#  
#  ITERATOR="--iteratorSuffix=ca"
#  CONTROLSTATUS="--controlSet"
#  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $LAZINESS $TIMEWINDOWOPTS
#  #
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  # python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  
#  ITERATOR="--iteratorSuffix=cb"
#  CONTROLSTATUS=""
#  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#  #
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  # python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  
#  ITERATOR="--iteratorSuffix=ccm"
#  CONTROLSTATUS=""
#  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#  #
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  # python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  
#  source ./shellScripts/calc_aligned_stim_preamble.sh
#  #
#  BLOCKSELECTOR="--blockIdx=1 --processAll"
#  #
#  ITERATOR="--iteratorSuffix=ccs"
#  CONTROLSTATUS=""
#  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS
#  #
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  # python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
#  

source ./shellScripts/calc_aligned_motion_preamble.sh
#
BLOCKSELECTOR="--blockIdx=2 --processAll"

iterators=(ca cb ccs ccm)
estimators=(mahal_ledoit)

TARGET="laplace_scaled"
# # for ITER in "${iterators[@]}"
# # do
# #   for EST in "${estimators[@]}"
# #   do
# #     python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
# #   done
# # done

TARGET="laplace_scaled"
# # for EST in "${estimators[@]}"
# # do
# #   python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
# # done

TARGET="laplace_spectral_scaled"
for ITER in "${iterators[@]}"
do
  for EST in "${estimators[@]}"
  do
    python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  done
done

TARGET="laplace_spectral_scaled"
for EST in "${estimators[@]}"
do
  python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done