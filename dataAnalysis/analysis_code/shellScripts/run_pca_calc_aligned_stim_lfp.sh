#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_stim_lapl

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-plots_stim_lapl.out
#SBATCH -e ../../batch_logs/%j-%a-plots_stim_lapl.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_stim_preamble.sh

# python -u './calcPCAinChunks.py' --inputBlockSuffix="kcsd" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK
python -u './calcSparsePCA.py' --inputBlockSuffix="lfp" --unitQuery="lfp" --estimatorName="pcal" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK
python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="lfp" --estimatorName="pcal" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
