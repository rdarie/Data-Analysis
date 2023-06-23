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
#SBATCH -J s03_assemble_ma_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s03_assemble_ma_202101_20.out
#SBATCH -e ../../batch_logs/covariance/s03_assemble_ma_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_27 201902_03)
# # 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202101_20)
for A in "${exps[@]}"
do
  echo "step 3 assemble dataframes, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  #
  ITERATOR="ma"
  ESTIMATOR='mahal_ledoit'
  #
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  # --preScale
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="lfp" --selectionName="lfp" $COMMONOPTS
  # python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_spectral" --selectionName="lfp_spectral" $COMMONOPTS
  python -u './assembleDataFrames.py'            --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  # python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_${ESTIMATOR}" --selectionName="lfp_${ESTIMATOR}" $COMMONOPTS
  # python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_spectral_${ESTIMATOR}" --selectionName="lfp_spectral_${ESTIMATOR}" $COMMONOPTS
done