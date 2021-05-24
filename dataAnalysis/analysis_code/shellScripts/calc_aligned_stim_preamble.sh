#!/bin/bash

source ./shellScripts/run_align_stim_preamble.sh

# LAZINESS="--lazy"
LAZINESS=""
#
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=-500 --winStop=1500"
#
# ALIGNQUERYTERM="stimOn"
ALIGNQUERYTERM="stimOnHighRate"
#
ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"