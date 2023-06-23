#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s02c_get_tts_control_201901_25

# Specify an output files
#SBATCH -o ../../batch_logs/covariance/s02c_get_tts_control_201901_25-%a.out
#SBATCH -e ../../batch_logs/covariance/s02c_get_tts_control_201901_25-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

exps=(201901_25)
for A in "${exps[@]}"
do
  echo "step 11 calc mahal dist, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  ####################
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
  #
  ITERATOR="--iteratorSuffix=ma"
  #
  ESTIMATOR='mahal_ledoit'
  #
  CONTROLSTATUS="--controlSet"
  ALIGNQUERYTERM="startingNoStim"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  # sleep 4 minutes to wait for step02b to reset the H5 file
  sleep 240
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  #
  ## ## python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ## ## python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_spectral_${ESTIMATOR}" --unitQuery="mahal" --selectionName="lfp_spectral_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName="rig" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp" --unitQuery="lfp" --selectionName="lfp" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  ## ## python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="lfp_spectral" --unitQuery="lfp" --selectionName="lfp_spectral" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done