#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plotsStim

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsStim.stdout
#SBATCH -e ../../batch_logs/%j-plotsStim.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

python3 -u './plotAlignedNeurons.py' --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $UNITSELECTOR $WINDOW --unitQuery="all" $ALIGNQUERY $ALIGNFOLDER --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="rig" --unitQuery="all" $ALIGNQUERY $ALIGNFOLDER --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK
# python3 -u './plotAlignedAsigs.py' --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW --inputBlockName="lfp" --unitQuery="lfp" $ALIGNQUERY $ALIGNFOLDER --enableOverrides $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK