#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20200318

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignStim_20200318.stdout
#SBATCH -e ../batch_logs/%j-alignStim_20200318.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
EXP="exp202004271200"
# EXP="exp202003201200"

LAZINESS="--lazy"
# LAZINESS=""
# WINDOW="--window=miniRC"
# WINDOW="--window=short"
WINDOW="--window=extraShort"
# WINDOW="--window=extraExtraShort"
# TRIALSELECTOR="--blockIdx=2"
# TRIALSELECTOR="--blockIdx=3"
TRIALSELECTOR="--processAll"
# ANALYSISSELECTOR="--analysisName=emg"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=emg1msec"
# ANALYSISSELECTOR="--analysisName=emg1msecSmooth"
# ANALYSISSELECTOR="--analysisName=emg1msecNoLFPFilterSmoothEMG"
ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isispinaloremg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters $ANALYSISSELECTOR 
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes --chanQuery="isiemg" --blockName="emg" --verbose  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ANALYSISSELECTOR --eventName=stimAlignTimes --chanQuery="isispinal" --blockName="spinal" --verbose  --alignFolderName=stim

# python3 ./calcTrialOutliers.py --exp=$EXP --alignFolderName=stim --inputBlockName="lfp" $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW --alignQuery="stimOn" --saveResults --verbose --plotting
# python3 ./calcISIStimArtifact.py --exp=$EXP --alignFolderName=stim --inputBlockName="lfp" --outputBlockName="lfp_clean" $TRIALSELECTOR $ANALYSISSELECTOR $UNITSELECTOR $WINDOW --alignQuery="stimOn" --saveResults --verbose --plotting

# python3 ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR $ANALYSISSELECTOR $WINDOW --resultName="corr" --alignQuery="stimOn" --alignFolderName=stim --inputBlockName="lfp" --verbose --plotting

# python3 ./plotMatrixOfScalars.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --resultName="corr" --alignFolderName=stim --inputBlockName="lfp" --verbose
# python3 ./plotMatrixOfScalars.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --resultName="corrPreStim" --alignFolderName=stim --inputBlockName="lfp" --verbose
# python3 ./plotMatrixOfScalarsTopo.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --resultName="corr" --alignFolderName=stim --inputBlockName="lfp" --verbose
# python3 ./plotMatrixOfScalarsTopo.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --resultName="corrPreStim" --alignFolderName=stim --inputBlockName="lfp" --verbose
# python3 ./adhoc_plotMatrixOfScalarsTopoDarpaMAC7.py --exp=$EXP $TRIALSELECTOR $WINDOW $ANALYSISSELECTOR --alignFolderName=stim --inputBlockName="lfp" --verbose