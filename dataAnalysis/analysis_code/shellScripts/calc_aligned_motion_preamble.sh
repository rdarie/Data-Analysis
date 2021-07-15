#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

# LAZINESS="--lazy"
LAZINESS=""

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=-600 --winStop=1000"
#
# ALIGNQUERYTERM="startingNoStim"
ALIGNQUERYTERM="starting"
# ALIGNQUERYTERM="outbound"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"