#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s02a_get_tts_stim_201902

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s02a_get_tts_stim_201902-%a.out
#SBATCH -e ../../batch_logs/covariance/s02a_get_tts_stim_201902-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=5

exps=(201902_03 201902_04 201902_05)
for A in "${exps[@]}"
do
  echo "step 11 calc mahal dist, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  ####################
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
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
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="rig" --unitQuery="rig" --selectionName="rig" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="lfp" --unitQuery="lfp" --selectionName="lfp" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ## ## spython -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="lfp_spectral" --unitQuery="lfp" --selectionName="lfp_spectral" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ## ## python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="lfp_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ## ## python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="lfp_spectral_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_spectral_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done