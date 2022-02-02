#!/bin/bash

source ./shellScripts/run_align_stim_raw_preamble.sh

# LAZINESS="--lazy"
LAZINESS=""
#
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=-100 --winStop=100"
#
ALIGNQUERYTERM="stimOn"
# ALIGNQUERYTERM="stimOnHighRate"
##
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"