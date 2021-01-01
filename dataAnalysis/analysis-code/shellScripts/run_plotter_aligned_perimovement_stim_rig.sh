#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plotsStim_rig

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsStim_rig.stdout
#SBATCH -e ../../batch_logs/%j-plotsStim_rig.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_plotter_aligned_perimovement_stim_preamble.sh

python3 -u './plotAlignedAsigs.py' --inputBlockName="rig" --unitQuery="all" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $PAGELIMITS
