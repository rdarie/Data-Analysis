#!/bin/bash

source ./shellScripts/run_align_stim_preamble.sh

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

STATSOVERLAY="--overlayStats"
# STATSOVERLAY=""

TIMEWINDOWOPTS="--winStart=-400 --winStop=1000"

# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=stimOnLowRate"
ALIGNQUERY="--alignQuery=stimOnHighRate"
# ALIGNQUERY="--alignQuery=outbound"

# HUEOPTS="--hueName=amplitude --hueControl="
# ROWOPTS="--rowName=RateInHz --rowControl="
# COLOPTS="--colName=electrode --colControl=control"
# STYLEOPTS="--styleName= --styleControl="
# SIZEOPTS="--sizeName= --sizeControl="

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl="
COLOPTS="--colName=pedalMovementCat --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

OTHERASIGOPTS="--recalcStats"
OTHERNEURONOPTS=""
