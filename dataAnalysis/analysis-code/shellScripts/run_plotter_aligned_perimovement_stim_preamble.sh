#!/bin/bash

source ./shellScripts/run_align_perimovement_stim_preamble.sh

# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""

#STATSOVERLAY="--overlayStats"
TIMEWINDOWOPTS="--winStart=200 --winStop=800"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=outbound"

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl="
COLOPTS="--colName=electrode --colControl=control"
STYLEOPTS="--styleName=RateInHz"

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""
OTHERASIGOPTS=""
OTHERNEURONOPTS=""