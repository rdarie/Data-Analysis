#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J nix_assembly_20

# Specify an output file
#SBATCH -o ../../batch_logs/nix_assembly_20.out
#SBATCH -e ../../batch_logs/nix_assembly_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2

source ./shellScripts/run_exp_preamble_20.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# primary data
blocks=(lfp lfp_CAR rig)
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
    INPUTBLOCKNAME="--inputBlockSuffix=lfp"
    UNITQUERY="--unitQuery=lfp"
    python -u ./calcTrialOutliersV2.py --exp=$EXP $BLOCKSELECTOR $UNITSELECTOR $WINDOW --alignFolderName=$A $ANALYSISFOLDER $ALIGNQUERY $LAZINESS $UNITQUERY $INPUTBLOCKNAME --plotting --verbose --saveResults
done