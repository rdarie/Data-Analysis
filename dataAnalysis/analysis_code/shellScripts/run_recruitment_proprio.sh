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
#SBATCH -o ../../batch_logs/%j-plotsStim.out
#SBATCH -e ../../batch_logs/%j-plotsStim.out

# Specify account details
#SBATCH --account=carney-dborton-condo

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/calc_aligned_motion_preamble.sh

TIMEWINDOWOPTS="--winStart=-100 --winStop=400"
INPUTBLOCKNAME="--inputBlockSuffix=lfp_CAR_spectral_fa_mahal"
UNITSELECTOR="--unitQuery=mahal"

# ALIGNQUERYTERM="startingNoStim"
ALIGNQUERYTERM="starting"
# ALIGNQUERYTERM="outbound"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"

python -u "./calcSignalRecruitment.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY $TIMEWINDOWOPTS
# python -u "./plotRecruitment.py" --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
