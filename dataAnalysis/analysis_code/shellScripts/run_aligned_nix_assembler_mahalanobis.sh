#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J nix_assembly_mahal_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/nix_assembly_mahal_202101_20.out
#SBATCH -e ../../batch_logs/nix_assembly_mahal_202101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
# EE              SBATCH --array=2,3

SLURM_ARRAY_TASK_ID=2

source ./shellScripts/run_exp_preamble_202101_20.sh
source ./shellScripts/calc_aligned_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# primary data
blocks=(laplace laplace_spectral)
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
      # python -u ./assembleExperimentAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR --inputBlockSuffix=$B $WINDOW $ANALYSISFOLDER --alignFolderName=$A $LAZINESS
  done
  TIMEWINDOWOPTS="--winStart=-600 --winStop=150"
  PAGELIMITS="--limitPages=10"
  HUEOPTS="--hueName=trialAmplitude --hueControl="
  OUTLIERMASK="--maskOutlierBlocks"
  python -u ./plotAlignedAsigsTopo.py --inputBlockSuffix="laplace_spectral" --unitQuery="csd" --alignQuery="stimOnHighRate" --alignFolderName=$A $TIMEWINDOWOPTS $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, trialRateInHz, freqBandName" $HUEOPTS $OUTLIERMASK --exp=$EXP $WINDOW $ANALYSISFOLDER $BLOCKSELECTOR
  PAGELIMITS="--limitPages=2"
  python -u ./plotAlignedAsigsTopo.py --inputBlockSuffix="laplace" --unitQuery="csd" --alignQuery="stimOnHighRate" --alignFolderName=$A $TIMEWINDOWOPTS $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, trialRateInHz" $HUEOPTS $OUTLIERMASK --exp=$EXP $WINDOW $ANALYSISFOLDER $BLOCKSELECTOR
  done