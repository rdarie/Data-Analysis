#!/bin/bash


LAZINESS="--lazy"

WINDOWTERM="XXS"
WINDOW="--window=${WINDOWTERM}"
ANALYSISFOLDER="--analysisName=fullRes"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=stim"
#
ALIGNFOLDER="--alignFolderName=stim"