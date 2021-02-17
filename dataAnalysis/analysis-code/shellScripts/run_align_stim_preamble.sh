#!/bin/bash

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp202101141100"
# EXP="exp202101191100"
EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"

LAZINESS="--lazy"
# LAZINESS=""

# WINDOW="--window=long"
WINDOW="--window=M"
AMPFIELDNAME="--amplitudeFieldName=amplitude"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"
# ANALYSISFOLDER="--analysisName=normalizedByImpedance"

# SIGNALFOLDER="--signalSubfolder=loRes"
SIGNALFOLDER="--signalSubfolder=default"
# SIGNALFOLDER="--signalSubfolder=normalizedByImpedance"

EVENTFOLDER="--eventSubfolder=None"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=stim"
#
# ALIGNFOLDER="--alignFolderName=motion"
ALIGNFOLDER="--alignFolderName=stim"