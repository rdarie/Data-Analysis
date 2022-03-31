#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s03a_assemble_normalize_202101_22

# Specify an output file
#SBATCH -o ../../batch_logs/s03a_assemble_normalize_202101_22.out
#SBATCH -e ../../batch_logs/s03a_assemble_normalize_202101_22.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

#    SLURM_ARRAY_TASK_ID=2
#

# exps=(201902_03 201902_04 201902_05)
# exps=(201901_25 201901_26 201901_27 201902_03 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_22)
for A in "${exps[@]}"
do
  echo "step 03 assemble plot normalize, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="pa"
  #
  # python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  #
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  # python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="lfp" --selectionName="lfp" $COMMONOPTS
  # python -u './assembleDataFrames.py' --inputBlockSuffix="laplace" --selectionName="laplace" $COMMONOPTS
  # python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral" --selectionName="laplace_spectral" $COMMONOPTS
  # python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  #
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="na"
  #
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  #
  python -u './assembleDataFrames.py' --resetHDF --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace' --selectionName='laplace' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_spectral' --selectionName='laplace_spectral' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  #
  TARGET="laplace"
  python -u './calcSignalNormalization.py' --estimatorName="scaled" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  #
  TARGET="laplace_spectral"
  python -u './calcSignalNormalization.py' --estimatorName="scaled" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
done