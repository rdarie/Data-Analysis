#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J s13c_apply_mahal_dist_stim_202102_02

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s13c_apply_mahal_dist_stim_202101_02-%a.out
#SBATCH -e ../../batch_logs/covariance/s13c_apply_mahal_dist_stim_202101_02-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1

# SLURM_ARRAY_TASK_ID=1

# 201902_03 201902_04 201902_05
# exps=(201901_25 201901_26 201901_27 201902_03 201902_04 201902_05
# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202102_02)
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
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace_scaled_${ESTIMATOR}" --unitQuery="mahal" --selectionName="laplace_scaled_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral_scaled_${ESTIMATOR}" --unitQuery="mahal" --selectionName="laplace_spectral_scaled_${ESTIMATOR}" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_scaled" --unitQuery="laplace" --selectionName="laplace_scaled" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral_scaled" --unitQuery="laplace" --selectionName="laplace_spectral_scaled" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig" --unitQuery="rig" --selectionName="rig" --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done