#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J s03b_prelim_plots_202101_21

# Specify an output file
#SBATCH -o ../../batch_logs/s03b_prelim_plots_202101_21.out
#SBATCH -e ../../batch_logs/s03b_prelim_plots_202101_21.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

#    SLURM_ARRAY_TASK_ID=2
#

# exps=(201902_03 201902_04 201902_05)
# exps=(201901_25 201901_26 201901_27 201902_03 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_21)
for A in "${exps[@]}"
do
  echo "step 03b plots, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="pa"
  #
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  #
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="laplace" --selectionName="laplace" $COMMONOPTS
  python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral" --selectionName="laplace_spectral" $COMMONOPTS
  python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  #
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="laplace" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="rig_illustration" --verbose=1 --selectionName="rig" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="laplace_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
done