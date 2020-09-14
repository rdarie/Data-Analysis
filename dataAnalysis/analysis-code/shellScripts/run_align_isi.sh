#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J alignStim_20200701

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignStim_20200701.stdout
#SBATCH -e ../../batch_logs/%j-alignStim_20200701.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202003201200"
# EXP="exp202006171300"
EXP="exp202007011300"
# EXP="exp202007021300"
# EXP="exp202007071300"
# EXP="exp202007081300"
# EXP="exp202009031500"

LAZINESS="--lazy"
# LAZINESS=""

# WINDOW="--window=XXS"
WINDOW="--window=XS"
# WINDOW="--window=XSPre"

SLURM_ARRAY_TASK_ID=1
# TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# TRIALSELECTOR="--blockIdx=3"
TRIALSELECTOR="--processAll"

# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=hiRes"
ANALYSISSELECTOR="--analysisName=loRes"
# ANALYSISSELECTOR="--analysisName=fullRes"
#
UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinal"
# UNITSELECTOR="--unitQuery=isiemg"
# UNITSELECTOR="--unitQuery=isiemgoracc"
# UNITSELECTOR="--unitQuery=isispinaloremg"

OUTPUTBLOCKNAME="--outputBlockName=emg_clean"
INPUTBLOCKNAME="--inputBlockName=emg"
# OUTPUTBLOCKNAME="--outputBlockName=lfp_clean"
# INPUTBLOCKNAME="--inputBlockName=lfp"
# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=all"

# CHANSELECTOR="--chanQuery=isiemg"
# CHANSELECTOR="--chanQuery=isispinal"

python3 -u ./assembleExperimentData.py --exp=$EXP --blockIdx=3 --processAsigs --processRasters $ANALYSISSELECTOR 

CHANSELECTOR="--chanQuery=all"
OUTPUTBLOCKNAME="--outputBlockName=all"
python3 -u ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes $CHANSELECTOR $OUTPUTBLOCKNAME --verbose --alignFolderName=stim
WINDOW="--window=XSPre"
python3 -u ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes $CHANSELECTOR $OUTPUTBLOCKNAME --verbose --alignFolderName=stim
