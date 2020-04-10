#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20200309

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignStim_20200309.stdout
#SBATCH -e ../batch_logs/%j-alignStim_20200309.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp202003091200"
EXP="exp202003201200"

LAZINESS="--lazy"
# LAZINESS=""
# WINDOW="--window=miniRC"
WINDOW="--window=short"
# WINDOW="--window=extraShort"
# TRIALSELECTOR="--blockIdx=2"
TRIALSELECTOR="--processAll"
# ANALYSISSELECTOR="--analysisName=emg"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
ANALYSISSELECTOR="--analysisName=emgHiRes"
#
#UNITSELECTOR="--unitQuery=all"
UNITSELECTOR="--unitQuery=isiemg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters $ANALYSISSELECTOR 
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes --chanQuery="all" --blockName="lfp"  --alignFolderName=stim
# python3 ./calcAlignedRasters.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes --chanQuery="all"  --alignFolderName=stim
# python3 ./calcTrialOutliers.py --exp=$EXP --alignFolderName=stim --inputBlockName="lfp" $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW --plotting --alignQuery="all" --verbose