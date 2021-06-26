#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=16
#SBATCH --mem-per-cpu=32G

# Specify a job name:
#SBATCH -J ols_motion_lfp_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ols_motion_lfp_20.out
#SBATCH -e ../../batch_logs/%j-%a-ols_motion_lfp_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="startingNoStim"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
ITERATOR="a"
WINDOWTERM="XL"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_c --selectionNameRhs=lfp_CAR"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_c --selectionNameLhs=rig"
###
TARGET="lfp_CAR"
ESTIMATOR="pca"
###
# iterators=(a b c d e f)
iterators=(g)
for ITER in "${iterators[@]}"
do
    echo "On iterator $ITER"
    # python -u './createToyDataFromDataFrames.py' --iteratorSuffix=$ITER --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
    # python -u './calcGridSearchSignalDimensionality.py' --estimatorName=$ESTIMATOR --datasetName="Synthetic_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
    # python -u './processSignalDimensionality.py' --estimatorName=$ESTIMATOR --datasetName="Synthetic_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
    #
    # python -u './calcGridSearchSignalDimensionality.py' --averageByTrial --estimatorName="${ESTIMATOR}_ta" --datasetName="Synthetic_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
    # python -u './processSignalDimensionality.py' --estimatorName="${ESTIMATOR}_ta" --datasetName="Synthetic_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting
done
###
# python -u './compareSignalCovarianceMatrices.py' --estimatorName=$ESTIMATOR --iteratorSuffixList="a, b, c, d, e, f" --datasetPrefix="Synthetic_${WINDOWTERM}_df" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=1 --plotting


ITERATOR="g"
WINDOWTERM="XL"
RHSOPTS="--datasetNameRhs=Synthetic_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=lfp_CAR"
LHSOPTS="--datasetNameLhs=Synthetic_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"

python -u './calcGridSearchRegressionWithPipelines.py' --transformerNameRhs='pca' --debugging --estimatorName='enr' --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
python -u './calcGridSearchRegressionWithPipelines.py' --transformerNameRhs='pca_ta' --debugging --estimatorName='enr' --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2