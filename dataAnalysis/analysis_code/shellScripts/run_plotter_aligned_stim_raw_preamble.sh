#!/bin/bash

source ./shellScripts/run_align_stim_preamble.sh
WINDOW="--window=XS"
ANALYSISFOLDER="--analysisName=fullRes"

# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""

# STATSOVERLAY="--overlayStats"

STATSOVERLAY=""
TIMEWINDOWOPTS="--winStart=22 --winStop=2"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=stimOnHighRate"
# ALIGNQUERY="--alignQuery=outbound"

# HUEOPTS="--hueName=amplitude --hueControl="
# ROWOPTS="--rowName=RateInHz --rowControl="
# COLOPTS="--colName=electrode --colControl=control"
# STYLEOPTS="--styleName= --styleControl="
# SIZEOPTS="--sizeName= --sizeControl="

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl="
COLOPTS="--colName= --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=10"
PAGELIMITS=""

OTHERASIGOPTS="--recalcStats"
OTHERNEURONOPTS=""
