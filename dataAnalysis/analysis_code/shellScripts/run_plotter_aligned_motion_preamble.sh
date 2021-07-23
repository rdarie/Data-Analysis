#!/bin/bash

source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/run_align_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"


OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

STATSOVERLAY="--overlayStats"
# STATSOVERLAY=""

TIMEWINDOWOPTS="--winStart=-300 --winStop=700"

# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=outbound"
ALIGNQUERY="--alignQuery=starting"
# ALIGNQUERY="--alignQuery=stopping"
# ALIGNQUERY="--alignQuery=startingNoStim"

HUEOPTS="--hueName=trialAmplitude --hueControl="
ROWOPTS="--rowName=pedalMovementCat --rowControl="
COLOPTS="--colName=electrode --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# HUEOPTS="--hueName=expName --hueControl="
# ROWOPTS="--rowName=pedalMovementCat --rowControl="
# COLOPTS="--colName=pedalSizeCat --colControl="
# STYLEOPTS="--styleName= --styleControl="
# SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

# OTHERASIGOPTS="--noStim"
OTHERASIGOPTS="--noStim --recalcStats"
OTHERNEURONOPTS="--noStim"