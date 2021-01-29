#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j_%a-plots_motion_lfp.out
#SBATCH -e ../../batch_logs/%j_%a-plots_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=3

SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

python3 -u './plotAlignedAsigs.py' --inputBlockName="lfp" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS