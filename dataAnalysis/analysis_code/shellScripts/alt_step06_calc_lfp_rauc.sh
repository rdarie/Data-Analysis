#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=300G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J s06_mahal_dist_rauc_201901_27

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s06_mahal_dist_rauc_201901_27.out
#SBATCH -e ../../batch_logs/covariance/s06_mahal_dist_rauc_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_27 201902
# # 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202101_20)
for A in "${exps[@]}"
do
  echo "step 12 lfp rauc, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="ma"
  ESTIMATOR='mahal_ledoit'
  #
  ## ## TARGET="lfp_${ESTIMATOR}"
  ## ## INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  ## ## python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  ## ## #
  ## ## TARGET="lfp_spectral_${ESTIMATOR}"
  ## ## INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  ## ## python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  #
  TARGET="lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  #
  TARGET="lfp_spectral"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
done