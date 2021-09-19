#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J nix_assembly_27

# Specify an output file
#SBATCH -o ../../batch_logs/nix_assembly_27.out
#SBATCH -e ../../batch_logs/nix_assembly_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2

source ./shellScripts/run_exp_preamble_27.sh
source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# primary data
blocks=(lfp lfp_CAR rig)
# lfp_CAR_spectral
alignfolders=(stim)
# alignfolders=(motion)
#
for A in "${alignfolders[@]}"
do
  echo "    concatenating $A"
  for B in "${blocks[@]}"
  do
      echo "concatenating $B blocks"
      python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER --alignFolderName=$A $LAZINESS
  done
  python -u ./calcTrialOutliersV2.py --inputBlockSuffix=lfp --unitQuery=lfp --alignFolderName=$A $UNITSELECTOR $LAZINESS --plotting --verbose --saveResults --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $BLOCKSELECTOR
  TIMEWINDOWOPTS="--winStart=-100 --winStop=300"
  PAGELIMITS=""
  HUEOPTS="--hueName=trialAmplitude --hueControl="
  OUTLIERMASK="--maskOutlierBlocks"
  python -u ./plotAlignedAsigsTopo.py --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --alignQuery="stimOnHighRate" --alignFolderName=$A $TIMEWINDOWOPTS $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection" $HUEOPTS $OUTLIERMASK --exp=$EXP $WINDOW $ANALYSISFOLDER $BLOCKSELECTOR
done