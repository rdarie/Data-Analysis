#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J tts_assembler_plots_01_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/test_train_split_assembler_plots_01_202101_20.out
#SBATCH -e ../../batch_logs/test_train_split_assembler_plots_01_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
ITERATOR="pa"
#

# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_20)
for A in "${exps[@]}"
do
  echo "    concatenating $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  # --preScale
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  python -u './assembleDataFrames.py' --inputBlockSuffix="lfp" --resetHDF --selectionName="lfp" $COMMONOPTS
  python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  #
  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  python -u './plotFinalSynchConfirmation.py' --plotSuffix="final_synch_confirmation" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --winStart="-250" --winStop="750" --verbose=1 --selectionName=rig $OPTS
done