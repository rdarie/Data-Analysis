#!/bin/bash


LAZINESS="--lazy"
# LAZINESS=""
# VERBOSITY="--verbose"
VERBOSITY=""
WINDOWTERM="XL"
#
WINDOW="--window=${WINDOWTERM}"

ANALYSISFOLDER="--analysisName=hiRes"
# ANALYSISFOLDER="--analysisName=hiResHiFreq"
# ANALYSISFOLDER="--analysisName=default"
# 
SIGNALFOLDER="--signalSubfolder=hiRes"
# SIGNALFOLDER="--signalSubfolder=hiResHiFreq"
# SIGNALFOLDER="--signalSubfolder=default"

EVENTFOLDER="--eventSubfolder=None"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=stim"
#
ALIGNFOLDER="--alignFolderName=stim"