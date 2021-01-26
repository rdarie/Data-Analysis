#!/bin/bash

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202010011100"
# EXP="exp202012121100"
# EXP="exp202012171200"
EXP="exp202101061100"
EXP="exp202101141100"
EXP="exp202101191100"
EXP="exp202101201100"
EXP="exp202101211100"
EXP="exp202101251100"

LAZINESS="--lazy"

WINDOW="--window=M"
ANALYSISFOLDER="--analysisName=default"

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# BLOCKSELECTOR="--processAll"

EVENTSELECTOR="--eventName=motionAlignTimes"
#
ALIGNFOLDER="--alignFolderName=motion"