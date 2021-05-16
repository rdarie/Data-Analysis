#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=-500 --winStop=1500"

# ALIGNQUERYTERM="startingNoStim"
ALIGNQUERYTERM="starting"
# ALIGNQUERYTERM="outbound"
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"