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
#SBATCH -J s10_calc_baselines_202101_25

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s10_calc_baselines_202101_25.out
#SBATCH -e ../../batch_logs/covariance/s10_calc_baselines_202101_25.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# exps=(201901_27 201902 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(201901_25)
for A in "${exps[@]}"
do
  echo "step 10 calc baselines, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  
  ITERATOR="--iteratorSuffix=cd"
  CONTROLSTATUS="--controlSet"
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $LAZINESS $TIMEWINDOWOPTS
  #
  python -u './assembleDataFrames.py' $ITERATOR --resetHDF --inputBlockSuffix='laplace' --selectionName='laplace' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral' --selectionName='laplace_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_scaled' --selectionName='laplace_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='laplace_spectral_scaled' --selectionName='laplace_spectral_scaled' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  # python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='pedalState' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' $ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  #
  ITERATOR="cd"
  TARGET="laplace"
  python -u './calcSignalNormalization.py' --estimatorName="baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  #
  TARGET="laplace_spectral"
  python -u './calcSignalNormalization.py' --estimatorName="baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
done