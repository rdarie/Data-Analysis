#!/bin/bash


LAZINESS="--lazy"
# LAZINESS=""
VERBOSITY="--verbose"

WINDOWTERM="XL"
#
WINDOW="--window=${WINDOWTERM}"


ANALYSISFOLDER="--analysisName=hiRes"
# ANALYSISFOLDER="--analysisName=default"
# 
SIGNALFOLDER="--signalSubfolder=hiRes"
# SIGNALFOLDER="--signalSubfolder=default"

EVENTFOLDER="--eventSubfolder=None"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=stim"
#
ALIGNFOLDER="--alignFolderName=stim"