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
#SBATCH -J s05_spectral_prep_regression_stb_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/regression/s05_spectral_prep_regression_stb_202101_27.out
#SBATCH -e ../../batch_logs/regression/s05_spectral_prep_regression_stb_202101_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_27)
export CCV_HEADLESS=1
for A in "${exps[@]}"
do
  echo "step 05 pls regression preparation, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  
  ITERATOR="rb"
  
  # #python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  # ## --preScale
  # #python -u './assembleDataFrames.py' --resetHDF --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_baseline' --selectionName='laplace_baseline' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  # #python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='laplace_spectral_baseline' --selectionName='laplace_spectral_baseline' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  # #python -u './assembleDataFrames.py' --iteratorSuffix=$ITERATOR --inputBlockSuffix='rig' --selectionName='rig' --loadFromFrames --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
  #
  # OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  # python -u './plotFinalSynchConfirmation.py' --plotSuffix="final_synch_confirmation_baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --winStart="-300" --winStop="1000" --verbose=1 --selectionName="rig" --selectionName2="laplace_baseline" $OPTS
  # python -u './plotFinalSynchConfirmation.py' --plotSuffix="final_synch_confirmation_spectral_baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --winStart="-300" --winStop="1000" --verbose=1 --selectionName="rig" --selectionName2="laplace_spectral_baseline" $OPTS
  #
  ESTIMATOR="select"
  TARGET="laplace_spectral_baseline"
  #  --averageByTrial
  python -u './calcSignalColumnSelector.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  #
  ############################################################################################################
  RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=${TARGET}"
  LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
  #
  python -u './prepSignalsAsRegressorV3.py' --transformerNameRhs=${ESTIMATOR} --debugging --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=1
done