#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=8
#SBATCH --ntasks-per-core=8
#SBATCH --mem-per-cpu=96G

# Specify a job name:
#SBATCH -J pca_dimen_motion_lfp_ra_ta_25

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_dimen_motion_lfp_ra_ta_25.out
#SBATCH -e ../../batch_logs/%j-%a-pca_dimen_motion_lfp_ra_ta_25.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
TARGET="lfp_CAR_spectral"
# TARGET="lfp_CAR"
#
# ESTIMATOR="fa"
ESTIMATOR="pca"
#
#  iterators=(a b cs cm)
#  for ITER in "${iterators[@]}"
#  do
#    python -u './calcGridSearchSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#    python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
#    python -u './calcSignalNovelty.py' --estimatorName="mahal" --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
#  done
#
#  python -u './compareSignalCovarianceMatrices.py' --estimatorName="mahal" --iteratorSuffixList="a, b, cm, cs" --datasetPrefix="Block_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting

##################################################
# ITERATOR="a"
# ALIGNQUERYTERM="outbound"
##################################################
# ITERATOR="b"
# ALIGNQUERYTERM="startingNoStim"
##################################################
# ITERATOR="c"
# ALIGNQUERYTERM="starting"
##################################################
ITERATOR="ra"
ALIGNQUERYTERM="starting"
##################################################
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

#####
# python -u './testSignalNormality.py' --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
TARGET="lfp_CAR"
ESTIMATOR="pca_ta"
python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
ESTIMATOR="fa_ta"
python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting

TARGET="lfp_CAR_spectral"
ESTIMATOR="pca_ta"
python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
ESTIMATOR="fa_ta"
python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
