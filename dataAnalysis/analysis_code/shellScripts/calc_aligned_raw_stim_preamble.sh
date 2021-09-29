#!/bin/bash

source ./shellScripts/run_align_raw_stim_preamble.sh

# LAZINESS="--lazy"
LAZINESS=""
#
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=-200 --winStop=400"
#
ALIGNQUERYTERM="stimOn"
##
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"