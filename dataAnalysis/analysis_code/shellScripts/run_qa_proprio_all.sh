#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J qa_all_201902-03-05

# Specify an output file
#SBATCH -o ../../batch_logs/qa_all_201902-03-05-%a.out
#SBATCH -e ../../batch_logs/qa_all_201902-03-05-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
# Request custom resources
#SBATCH --array=4
#SBATCH --export=CCV_HEADLESS=1

# SLURM_ARRAY_TASK_ID=1
exps=(201902_03 201902_04 201902_05)
for A in "${exps[@]}"
do
  echo "step 01 qa, on $A"
  source shellScripts/run_exp_preamble_$A.sh
  source shellScripts/run_align_stim_preamble.sh
  ALIGNQUERY="--alignQuery=stimOn"
  UNITQUERY="--unitQuery=lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=lfp"
  python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
  
  source shellScripts/run_align_motion_preamble.sh
  ALIGNQUERY="--alignQuery=starting"
  UNITQUERY="--unitQuery=lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=lfp"
  python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --amplitudeFieldName="amplitude" --saveResults
