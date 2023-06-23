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
#SBATCH -J s08_calc_pca_201901_27

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s08_calc_pca_201901_27.out
#SBATCH -e ../../batch_logs/covariance/s08_calc_pca_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# exps=(201901_27 201902 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(201901_27)
for A in "${exps[@]}"
do
  echo "step 10 calc baselines, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"

  ITERATOR="ma"
  TARGET="lfp"
  ESTIMATOR='ica'
  python -u './calcSignalDimensionalityReduction.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting

  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="factor_illustration" --verbose=1 --selectionName="${TARGET}_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
done