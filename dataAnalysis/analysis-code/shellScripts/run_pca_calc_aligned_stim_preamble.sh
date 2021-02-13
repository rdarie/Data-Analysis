#!/bin/bash

source ./shellScripts/run_align_stim_preamble.sh

# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""


TIMEWINDOWOPTS="--winStart=200 --winStop=800"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=stimOnHighRate"
# ALIGNQUERY="--alignQuery=outbound"
