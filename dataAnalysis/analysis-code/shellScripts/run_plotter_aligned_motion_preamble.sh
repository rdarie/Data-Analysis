#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

# OUTLIERMASK="--maskOutlierBlocks"
OUTLIERMASK=""

#STATSOVERLAY="--overlayStats"
TIMEWINDOWOPTS="--winStart=200 --winStop=800"

# ALIGNQUERY="--alignQuery=stimOn"
ALIGNQUERY="--alignQuery=outbound"

HUEOPTS="--hueName= --hueControl="
ROWOPTS="--rowName=pedalDirection --rowControl="
COLOPTS="--colName= --colControl="
STYLEOPTS="--styleName= --styleControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

OTHERASIGOPTS="--noStim"
OTHERNEURONOPTS="--noStim"