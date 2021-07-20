#!/bin/bash

source shellScripts/run_exp_preamble.sh
LAZINESS="--lazy"
# LAZINESS=""

VERBOSITY="--verbose"

WINDOWTERM="XL"
#
WINDOW="--window=${WINDOWTERM}"

# ANALYSISFOLDER="--analysisName=default"
ANALYSISFOLDER="--analysisName=hiRes"

# SIGNALFOLDER="--signalSubfolder=default"
SIGNALFOLDER="--signalSubfolder=hiRes"

EVENTFOLDER="--eventSubfolder=None"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"
# BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
EVENTSELECTOR="--eventName=motion"
ALIGNFOLDER="--alignFolderName=motion"

