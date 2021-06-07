#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J nix_assembly

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-nix_assembly.out
#SBATCH -e ../../batch_logs/%j-%a-nix_assembly.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
# EE              SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

blocks=(lfp_CAR lfp_CAR_spectral rig)
# blocks=(lfp_CAR_spectral_fa lfp_CAR_spectral_fa_mahal)
alignfolders=(stim motion)
for B in "${blocks[@]}"
do
    echo "concatenating $B blocks"
    for A in "${alignfolders[@]}"
    do
      echo "    concatenating $A"
      python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER --alignFolderName=$A $LAZINESS
    done
done