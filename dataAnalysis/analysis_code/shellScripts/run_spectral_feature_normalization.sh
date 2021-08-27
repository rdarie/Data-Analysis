#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J spectral_normalize_27

# Specify an output file
#SBATCH -o ../../batch_logs/spectral_normalize_27.out
#SBATCH -e ../../batch_logs/spectral_normalize_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2


#   SBATCH --ntasks=4
#   SBATCH --ntasks-per-core=4
#   SBATCH --mem-per-cpu=64G
#  SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=2 --processAll"
#
ITERATOR=na
TARGET=lfp_CAR_spectral
python -u './calcSignalNormalization.py' --estimatorName="scaled" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
ITERATOR=na
TARGET=lfp_CAR
python -u './calcSignalNormalization.py' --estimatorName="scaled" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
