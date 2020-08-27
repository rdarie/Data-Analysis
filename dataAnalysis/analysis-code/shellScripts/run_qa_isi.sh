#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=6
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J QA_Stim_20200702

# Specify an output file
#SBATCH -o ../../batch_logs/%j-QA_Stim_20200702.stdout
#SBATCH -e ../../batch_logs/%j-QA_Stim_20200702.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

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

LAZINESS="--lazy"
# LAZINESS=""

# WINDOW="--window=XXS"
WINDOW="--window=XS"

# TRIALSELECTOR="--blockIdx=1"
# TRIALSELECTOR="--blockIdx=3"
TRIALSELECTOR="--processAll"

# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=loRes"
ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isiemgoracc"
# UNITSELECTOR="--unitQuery=isiacc"
# UNITSELECTOR="--unitQuery=isispinaloremg"

OUTPUTBLOCKNAME="--outputBlockName=emg_clean"
INPUTBLOCKNAME="--inputBlockName=emg"
# OUTPUTBLOCKNAME="--outputBlockName=lfp_clean"
# INPUTBLOCKNAME="--inputBlockName=lfp"
ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=all"

CHANSELECTOR="--chanQuery=isiemgoracc"
# CHANSELECTOR="--chanQuery=isiacc"
# CHANSELECTOR="--chanQuery=isispinal"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

INPUTBLOCKNAME="--inputBlockName=emg"
OUTPUTBLOCKNAME="--outputBlockName=emg_clean"
python3 -u ./cleanISIData.py --exp=$EXP --alignFolderName=stim $OUTPUTBLOCKNAME $INPUTBLOCKNAME $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW $ALIGNQUERY --saveResults --verbose --plotting

# INPUTBLOCKNAME="--inputBlockName=emg_clean"
# python3 -u ./calcTrialOutliers.py --exp=$EXP --alignFolderName=stim $INPUTBLOCKNAME $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW $ALIGNQUERY --verbose --plotting --saveResults
# 
# INPUTBLOCKNAME="--inputBlockName=emg_clean"
# UNITSELECTOR="--unitQuery=isiemgenv"
# python -u ./calcTargetNoiseCeiling.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim $INPUTBLOCKNAME $UNITSELECTOR --maskOutlierBlocks $ALIGNQUERY --plotting