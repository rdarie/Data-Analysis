#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_motion_rig

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-plots_motion_rig.stdout
#SBATCH -e ../../batch_logs/%j-%a-plots_motion_rig.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=4

# SLURM_ARRAY_TASK_ID=3
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

python3 -u './plotAlignedAsigs.py' --inputBlockName="rig" --unitQuery="all" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $PAGELIMITS $OTHERASIGOPTS
