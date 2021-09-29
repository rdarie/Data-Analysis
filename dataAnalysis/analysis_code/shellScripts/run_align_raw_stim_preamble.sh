#!/bin/bash


LAZINESS="--lazy"

WINDOW="--window=XS"

ANALYSISFOLDER="--analysisName=fullRes"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=stim"
#
ALIGNFOLDER="--alignFolderName=stim"