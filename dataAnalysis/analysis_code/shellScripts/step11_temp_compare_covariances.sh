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
#SBATCH -J s11_compare_covariances_202101_21

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s11_compare_covariances_202101_21.out
#SBATCH -e ../../batch_logs/covariance/s11_compare_covariances_202101_21.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# exps=(201901_27 201902_03 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202101_21)
for A in "${exps[@]}"
do
  echo "step 10 compare covariances, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"

  iterators=(ca cb ccs ccm)
  estimators=(mahal_ledoit)
  TARGET="laplace_scaled"
  # python -u './compareSignalSampleSizes.py' --estimatorName="sampleCount" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # for ITER in "${iterators[@]}"
  # do
  #   for EST in "${estimators[@]}"
  #   do
  #     python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  #   done
  # done
  #
  # for EST in "${estimators[@]}"
  # do
  #   python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # done
  #
  TARGET="laplace_spectral_scaled"
  # python -u './compareSignalSampleSizes.py' --estimatorName="sampleCount" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # for ITER in "${iterators[@]}"
  # #do
  # #  for EST in "${estimators[@]}"
  # #  do
  # #    python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  # #  done
  # #done
  #
  # for EST in "${estimators[@]}"
  # do
  #   python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
  # done
  #
  # #ITERATOR="ca"
  # #TARGET="laplace"
  # #python -u './calcSignalNormalization.py' --estimatorName="baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  #
  # #TARGET="laplace_spectral"
  # #python -u './calcSignalNormalization.py' --estimatorName="baseline" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
done

#"exp202101201100, exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100"
estimators=(mahal_ledoit)
for EST in "${estimators[@]}"
do
  python -u './compareSignalCovarianceMatricesAcrossExp.py' --expList="exp202101201100, exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100" --targetList="laplace_spectral_scaled, laplace_scaled" --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done