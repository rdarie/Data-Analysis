#!/bin/bash

source ./shellScripts/run_align_motion_preamble.sh

OUTLIERMASK="--maskOutlierBlocks"
# OUTLIERMASK=""

# STATSOVERLAY="--overlayStats"
STATSOVERLAY=""

TIMEWINDOWOPTS="--winStart=150 --winStop=450"

# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=outbound"
ALIGNQUERY="--alignQuery=starting"

# HUEOPTS="--hueName= --hueControl="
# ROWOPTS="--rowName=pedalDirection --rowControl="
# COLOPTS="--colName= --colControl="
# STYLEOPTS="--styleName= --styleControl="
# SIZEOPTS="--sizeName= --sizeControl="

HUEOPTS="--hueName=amplitude --hueControl="
ROWOPTS="--rowName=RateInHz --rowControl="
COLOPTS="--colName=pedalMovementCat --colControl="
STYLEOPTS="--styleName= --styleControl="
SIZEOPTS="--sizeName= --sizeControl="

# PAGELIMITS="--limitPages=5"
PAGELIMITS=""

OTHERASIGOPTS="--noStim"
OTHERNEURONOPTS="--noStim"