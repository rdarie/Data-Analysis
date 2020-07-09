#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=12
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J alignStim_20200501

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignStim_20200501.stdout
#SBATCH -e ../../batch_logs/%j-alignStim_20200501.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
# EXP="exp202004271200"
# EXP="exp202004301200"
# EXP="exp202005011400"
# EXP="exp202003201200"
EXP="exp202006171300"

LAZINESS="--lazy"
# LAZINESS=""

# WINDOW="--window=XXS"
WINDOW="--window=XS"

# TRIALSELECTOR="--blockIdx=1"
# TRIALSELECTOR="--blockIdx=3"
TRIALSELECTOR="--processAll"

# ANALYSISSELECTOR="--analysisName=emgHiRes"
ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isiemgenv"
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isiemg"
# UNITSELECTOR="--unitQuery=isispinaloremg"

OUTPUTBLOCKNAME="--outputBlockName=emg_clean"
INPUTBLOCKNAME="--inputBlockName=emg"
# OUTPUTBLOCKNAME="--outputBlockName=lfp_clean"
# INPUTBLOCKNAME="--inputBlockName=lfp"
# ALIGNQUERY="--alignQuery=stimOn"
# ALIGNQUERY="--alignQuery=all"

CHANSELECTOR="--chanQuery=isiemg"
# CHANSELECTOR="--chanQuery=isispinal"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# python3 ./assembleExperimentData.py --exp=$EXP --blockIdx=3 --processAsigs --processRasters $ANALYSISSELECTOR 

# CHANSELECTOR="--chanQuery=isiemg"
# OUTPUTBLOCKNAME="--outputBlockName=emg"
# python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes $CHANSELECTOR $OUTPUTBLOCKNAME --verbose  --alignFolderName=stim

# INPUTBLOCKNAME="--inputBlockName=emg"
# OUTPUTBLOCKNAME="--outputBlockName=emg_clean"
# python3 ./cleanISIData.py --exp=$EXP --alignFolderName=stim $OUTPUTBLOCKNAME $INPUTBLOCKNAME $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW $ALIGNQUERY --saveResults --verbose --plotting

INPUTBLOCKNAME="--inputBlockName=emg_clean"
python3 ./calcTrialOutliers.py --exp=$EXP --alignFolderName=stim $INPUTBLOCKNAME $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW $ALIGNQUERY --verbose --plotting --saveResults
