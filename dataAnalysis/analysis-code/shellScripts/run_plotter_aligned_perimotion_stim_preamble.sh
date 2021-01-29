#!/bin/bash

source ./shellScripts/run_align_perimotion_stim_preamble.sh

# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""

STATSOVERLAY="--overlayStats"
TIMEWINDOWOPTS="--winStart=150 --winStop=450"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=outbound"

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl=0"
COLOPTS="--colName=electrode --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""
OTHERASIGOPTS="--recalcStats"
OTHERNEURONOPTS=""