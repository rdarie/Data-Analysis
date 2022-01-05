#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s04b_tts_covariance_motion_lfp_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s04b_tts_covariance_motion_lfp_202101_20-%a.out
#SBATCH -e ../../batch_logs/covariance/s04b_tts_covariance_motion_lfp_202101_20-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2-3

# exps=(201902_03 201902_04 201902_05)
# exps=(201901_25 201901_26 201901_27)
# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_20)
TRAINDATASET=""
# TRAINDATASET="--datasetExp=201901271000-Murdoc"
for A in "${exps[@]}"
do
  echo "step 04 apply normalization, motion,  get data for covariance, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  ITERATOR="--iteratorSuffix=ca"
  ALIGNQUERYTERM="outbound"
  CONTROLSTATUS="--controlSet"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  ###
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace_scaled" --unitQuery="laplace" --selectionName='laplace_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral_scaled" --unitQuery="laplace" --selectionName='laplace_spectral_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace" --unitQuery="laplace" --selectionName='laplace' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  ITERATOR="--iteratorSuffix=cd"
  ALIGNQUERYTERM="outbound"
  CONTROLSTATUS="--controlSet"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  ###
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace" --unitQuery="laplace" --selectionName='laplace' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  #     ###
  ITERATOR="--iteratorSuffix=cb"
  ALIGNQUERYTERM="startingNoStim"
  CONTROLSTATUS=""
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  ###
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace_scaled" --unitQuery="laplace" --selectionName='laplace_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral_scaled" --unitQuery="laplace" --selectionName='laplace_spectral_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace" --unitQuery="laplace" --selectionName='laplace' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  ITERATOR="--iteratorSuffix=ccm"
  ALIGNQUERYTERM="startingExp${A}"
  CONTROLSTATUS=""
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  ###
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace_scaled" --unitQuery="laplace" --selectionName='laplace_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral_scaled" --unitQuery="laplace" --selectionName='laplace_spectral_scaled' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace" --unitQuery="laplace" --selectionName='laplace' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  # python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="pedalState" --selectionName='pedalState' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
done