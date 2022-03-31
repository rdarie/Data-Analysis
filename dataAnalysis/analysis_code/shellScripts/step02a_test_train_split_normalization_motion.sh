#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J s02a_tts_norm_motion_202101_22

# Specify an output file
#SBATCH -o ../../batch_logs/s02a_test_train_split_normalization_motion_202101_22-%a.out
#SBATCH -e ../../batch_logs/s02a_test_train_split_normalization_motion_202101_22-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=1
  
# 201902_03 201902_04 201902_05
# exps=(201901_25 201901_26 201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_22)
for A in "${exps[@]}"
do
  echo "test train split motion norm $A"
  source shellScripts/run_exp_preamble_$A.sh
  source shellScripts/calc_aligned_motion_preamble.sh
  ITERATOR="--iteratorSuffix=na"
  ALIGNQUERYTERM="outbound"
  CONTROLSTATUS="--controlSet"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  ###
  python -u './calcTestTrainSplit.py' $CONTROLSTATUS --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' $ALIGNQUERY $ITERATOR --eventName='motion' --eventBlockSuffix='epochs' --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS $TIMEWINDOWOPTS
  ###
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS --resetHDF --inputBlockSuffix="laplace"          --unitQuery="laplace" --selectionName='laplace'          --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="laplace_spectral" --unitQuery="laplace" --selectionName='laplace_spectral' --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  python -u './applyTestTrainSplit.py' $CONTROLSTATUS            --inputBlockSuffix="rig"              --unitQuery="rig"     --selectionName='rig'              --verbose $ALIGNQUERY $ITERATOR --exp=$EXP $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done
# next, go to run_test_train_split_assembler_xxx.sh