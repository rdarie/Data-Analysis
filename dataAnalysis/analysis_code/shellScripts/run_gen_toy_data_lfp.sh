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
source shellScripts/run_exp_preamble.sh
source shellScripts/calc_aligned_motion_preamble.sh

ALIGNQUERYTERM="startingNoStim"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
ITERATOR="ma"
###
TARGET="laplace_scaled"
ESTIMATOR="pca"
###
WINDOWTERM="XL"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=${TARGET}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
DIMRED="select"
###  --trialAverageLFP
# python -u './plotStateSpace3D.py' --trialAverageLFP --iteratorSuffix=$ITERATOR --transformerNameRhs="${DIMRED}" --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './plotStateSpace3D.py' --iteratorSuffix=$ITERATOR --transformerNameRhs="${DIMRED}" --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2

ALIGNQUERYTERM="startingNoStim"
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
ITERATOR="rc"
###
TARGET="laplace_scaled"
ESTIMATOR="pca"
###
WINDOWTERM="XL"
RHSOPTS="--datasetNameRhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameRhs=${TARGET}"
LHSOPTS="--datasetNameLhs=Block_${WINDOWTERM}_df_${ITERATOR} --selectionNameLhs=rig"
# iterators=(a b c d e f)
iterators=(g)
for ITER in "${iterators[@]}"
do
    echo "On iterator $ITER"
    python -u './createToyDataFromDataFramesV2.py' --iteratorSuffix=$ITER --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
    # python -u './createToyDataFromDataFramesV2.py' --lowNoise --iteratorSuffix=$ITER --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
    # python -u './createToyDataFromDataFramesV2.py' --forceField --iteratorSuffix=$ITER --exp=$EXP $LHSOPTS $RHSOPTS $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
done


