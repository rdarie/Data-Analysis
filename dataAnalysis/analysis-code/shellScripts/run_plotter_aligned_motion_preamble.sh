#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

#STATSOVERLAY="--overlayStats"
TIMEWINDOWOPTS="--winStart=200 --winStop=800"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=outbound"

