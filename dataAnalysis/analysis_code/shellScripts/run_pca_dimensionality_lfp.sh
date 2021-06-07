#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=24
#SBATCH --ntasks-per-core=24
#SBATCH --mem-per-cpu=32G

# Specify a job name:
#SBATCH -J pca_dimen_motion_lfp_c

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_dimen_motion_lfp_c.out
#SBATCH -e ../../batch_logs/%j-%a-pca_dimen_motion_lfp_c.out

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
ESTIMATOR="fa"
# ESTIMATOR="pca"
#
ITERATOR="a"
ALIGNQUERYTERM="outbound"
##
# ITERATOR="b"
# ALIGNQUERYTERM="startingNoStim"
##
# ITERATOR="c"
# ALIGNQUERYTERM="startingE5"
##
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

#####
# python -u './testSignalNormality.py' --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
# python -u './calcGridSearchSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting

#
# python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
#
# python -u './assembleExperimentAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix="${TARGET}" $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS

TARGET="lfp_CAR_spectral_fa"
# TARGET="lfp_CAR_fa"
# TARGET="lfp_CAR_spectral_pca"
# TARGET="lfp_CAR_pca"
# python -u './calcSignalNovelty.py' --estimatorName="mahal" --datasetName="Block_${WINDOW}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting --showFigures