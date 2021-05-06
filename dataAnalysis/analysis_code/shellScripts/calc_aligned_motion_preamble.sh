#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=0 --winStop=500"

ALIGNQUERY="--alignQuery=starting"
# ALIGNQUERY="--alignQuery=startingNoStim"
