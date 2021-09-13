#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J compare_covariances_25

# Specify an output file
#SBATCH -o ../../batch_logs/compare_covariances_25.out
#SBATCH -e ../../batch_logs/compare_covariances_25.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2


#   SBATCH --ntasks=4
#   SBATCH --ntasks-per-core=4
#   SBATCH --mem-per-cpu=64G
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_temp.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

iterators=(ca cb ccs ccm ra)
estimators=(mahal_ledoit)

TARGET="lfp"
for ITER in "${iterators[@]}"
do
  for EST in "${estimators[@]}"
  do
    python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
  done
done

# TARGET="lfp_CAR"
# for ITER in "${iterators[@]}"
# do
#   for EST in "${estimators[@]}"
#   do
#     python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#   done
# done
#
# TARGET="lfp_CAR_spectral_scaled"
# for ITER in "${iterators[@]}"
# do
#   for EST in "${estimators[@]}"
#   do
#     python -u './calcSignalCovarianceMatrix.py' --estimatorName="${EST}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#   done
# done
#

TARGET="lfp_CAR"
for EST in "${estimators[@]}"
do
  python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done

# TARGET="lfp_CAR"
# for EST in "${estimators[@]}"
# do
#   python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
# done
# #
# TARGET="lfp_CAR_spectral_scaled"
# for EST in "${estimators[@]}"
# do
#   python -u './compareSignalCovarianceMatrices.py' --estimatorName="${EST}" --iteratorSuffixList="ca, cb, ccs, ccm" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
# done
