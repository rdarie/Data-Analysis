#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"


OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

STATSOVERLAY="--overlayStats"
# STATSOVERLAY=""

TIMEWINDOWOPTS="--winStart=-400 --winStop=1000"

# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=outbound"
# ALIGNQUERY="--alignQuery=starting"
# ALIGNQUERY="--alignQuery=stopping"
ALIGNQUERY="--alignQuery=startingNoStim"

# HUEOPTS="--hueName= --hueControl="
# ROWOPTS="--rowName=pedalSizeCat --rowControl="
# COLOPTS="--colName= --colControl="
# STYLEOPTS="--styleName= --styleControl="
# SIZEOPTS="--sizeName= --sizeControl="


HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=pedalSizeCat --rowControl="
COLOPTS="--colName=pedalMovementCat --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

# OTHERASIGOPTS="--noStim"
OTHERASIGOPTS="--noStim --recalcStats"
OTHERNEURONOPTS="--noStim"