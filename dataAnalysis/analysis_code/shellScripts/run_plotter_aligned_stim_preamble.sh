#!/bin/bash

source ./shellScripts/run_align_stim_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
#
# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""

STATSOVERLAY="--overlayStats"
# STATSOVERLAY=""

TIMEWINDOWOPTS="--winStart=-700 --winStop=700"

# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=stimOnLowRate"
ALIGNQUERY="--alignQuery=stimOnHighRate"
# ALIGNQUERY="--alignQuery=outbound"

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl="
COLOPTS="--colName=electrode --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

OTHERASIGOPTS="--recalcStats"
OTHERNEURONOPTS=""
