#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J s13a_apply_mahal_dist_stim_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s13a_apply_mahal_dist_stim_202101_27-%a.out
#SBATCH -e ../../batch_logs/covariance/s13a_apply_mahal_dist_stim_202101_27-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1

#  SLURM_ARRAY_TASK_ID=1
TRAINDATASET=""
# TRAINDATASET="--datasetExp=201902031100-Murdoc"

# 201902_03 201902_04 201902_05
# exps=(201901_25 201901_26 201901_27 201902_03 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 13 calc mahal dist, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  
  ####################
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
  ITERATOR="ca"
  #
  ALIGNQUERYTERM="stimOn"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  #
  targets=(laplace_scaled laplace_spectral_scaled)
  estimators=(mahal_ledoit)
  #
  for TARGET in "${targets[@]}"
  do
    for ESTIMATOR in "${estimators[@]}"
    do
      python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="${TARGET}" --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET $TRAINDATASET --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
      # python -u './makeViewableBlockFromTriggered.py' --plotting --inputBlockSuffix="${TARGET}_${ESTIMATOR}" --unitQuery="mahal" $VERBOSITY --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
    done
  done
  # redo outlier detection
  UNITQUERY="--unitQuery=mahal"
  INPUTBLOCKNAME="--inputBlockSuffix=laplace_scaled_mahal_ledoit"
  # # python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
  #
  ITERATOR="--iteratorSuffix=ma"
  #
  ALIGNQUERYTERM="stimOnExp${A}"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  CONTROLSTATUS=""
  ESTIMATOR='mahal_ledoit'
  #
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ROIOPTS $ITERATOR --eventName='stim' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace_scaled_${ESTIMATOR}" --unitQuery="mahal" --selectionName="laplace_scaled_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_spectral_scaled_${ESTIMATOR}" --unitQuery="mahal" --selectionName="laplace_spectral_scaled_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName="rig" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_scaled" --unitQuery="laplace" --selectionName="laplace_scaled" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_spectral_scaled" --unitQuery="laplace" --selectionName="laplace_spectral_scaled" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_baseline" --unitQuery="laplace" --selectionName="laplace_baseline" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="laplace_spectral_baseline" --unitQuery="laplace" --selectionName="laplace_spectral_baseline" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done