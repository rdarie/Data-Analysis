#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=300G

# Specify a job name:
#SBATCH -J lapl_calc_stim_raw_202101_21

# Specify an output file
#SBATCH -o ../../batch_logs/lapl_calc_stim_raw_202101_21-%a.out
#SBATCH -e ../../batch_logs/lapl_calc_stim_raw_202101_21-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1-3

# SLURM_ARRAY_TASK_ID=1
source ./shellScripts/run_exp_preamble_202101_21.sh
source ./shellScripts/calc_aligned_raw_stim_preamble.sh

echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
python -u ./calcLaplacianFromTriggered.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
#
# python -u ./calcWaveletFeatures.py --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS