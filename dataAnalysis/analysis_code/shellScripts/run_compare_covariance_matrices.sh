#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=4
#SBATCH --ntasks-per-core=4
#SBATCH --mem-per-cpu=72G

# Specify a job name:
#SBATCH -J compare_covariances_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-compare_covariances_27.out
#SBATCH -e ../../batch_logs/%j-%a-compare_covariances_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
TARGET="lfp_CAR"
#
iterators=(ca cb ccs ccm)
for ITER in "${iterators[@]}"
do
  python -u './calcSignalNovelty.py' --estimatorName="mahal" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
done

python -u './compareSignalCovarianceMatrices.py' --estimatorName="mahal" --iteratorSuffixList="ca, cb, ccm, ccs" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting

TARGET="lfp_CAR_spectral"
# TARGET="lfp_CAR"
#
iterators=(ca cb ccs ccm)
for ITER in "${iterators[@]}"
do
  python -u './calcSignalNovelty.py' --estimatorName="mahal" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
done

python -u './compareSignalCovarianceMatrices.py' --estimatorName="mahal" --iteratorSuffixList="ca, cb, ccm, ccs" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
