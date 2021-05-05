#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J test_train_split

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-test_train_split.out
#SBATCH -e ../../batch_logs/%j-%a-test_train_split.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#               SBATCH --array=2,3

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh
ITERATOR="--iteratorSuffix=b"
# python -u './calcTestTrainSplit.py' --blockIdx=2 --processAll $ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="limbState" --selectionName='limbState' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS

# blocks=(lfp_CAR lfp_CAR_spectral rig)
blocks=(lfp_CAR_spectral)
for B in "${blocks[@]}"
do
    echo "concatenating $B blocks"
    python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP --blockIdx=2 --processAll --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER $ALIGNFOLDER
done