#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J nix_assembly_mahal_28

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-nix_assembly_mahal_28.out
#SBATCH -e ../../batch_logs/%j-%a-nix_assembly_mahal_28.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
# EE              SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2

source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# derived without dim. reduction
blocks=(lfp_CAR_mahal lfp_CAR_spectral_mahal)
#
# alignfolders=(stim motion)
alignfolders=(motion)
#
for A in "${alignfolders[@]}"
do
    echo "    concatenating $A"
    for B in "${blocks[@]}"
    do
      echo "concatenating $B blocks"
      python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER --alignFolderName=$A $LAZINESS
    done
done