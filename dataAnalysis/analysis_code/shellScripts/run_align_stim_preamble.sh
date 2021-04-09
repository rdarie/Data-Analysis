#!/bin/bash
module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg fftw scons
module load anaconda/2020.02
module load mpi
# module load opengl
module load qt/5.10.1
module load zlib/1.2.11
module unload python

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
# EXP="exp202102151100"

LAZINESS="--lazy"
# LAZINESS=""

# WINDOW="--window=L"
WINDOW="--window=M"

AMPFIELDNAME="--amplitudeFieldName=amplitude"

# ANALYSISFOLDER="--analysisName=hiRes"
# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"
# ANALYSISFOLDER="--analysisName=normalizedByImpedance"

# SIGNALFOLDER="--signalSubfolder=hiRes"
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