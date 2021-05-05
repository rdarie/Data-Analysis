#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

TIMEWINDOWOPTS="--winStart=300 --winStop=1300"

ALIGNQUERY="--alignQuery=startingNoStim"
