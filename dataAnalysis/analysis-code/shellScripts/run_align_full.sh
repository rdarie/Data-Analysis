#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignFull_20190127.stdout
#SBATCH -e ../../batch_logs/%j-alignFull_20190127.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
EXP="exp201901271000"

LAZINESS="--lazy"

# WINDOW="--window=long"
WINDOW="--window=XS"

ANALYSISFOLDER="--analysisName=loRes"
# ANALYSISFOLDER="--analysisName=default"

SLURM_ARRAY_TASK_ID=4

# TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
TRIALSELECTOR="--processAll"
UNITSELECTOR="--selector=_minfrmaxcorr"

EVENTSELECTOR="--eventName=stimAlignTimes"
ALIGNFOLDER="--alignFolderName=stim"

# python3 -u ./assembleExperimentData.py --exp=$EXP $ANALYSISFOLDER --processAsigs --processRasters

python3 -u ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER --chanQuery="rig" --outputBlockName="rig" --verbose
python3 -u ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER --chanQuery="fr" --outputBlockName="fr" --verbose
python3 -u ./calcAlignedRasters.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW $EVENTSELECTOR $ALIGNFOLDER $ANALYSISFOLDER --chanQuery="raster" --outputBlockName="raster" --verbose

# qa
# python3 -u ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
# python3 -u ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting

# python3 -u ./selectUnitsByMeanFRandCorrelationAndAmplitude.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --verbose
# or
# python3 -u ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --verbose

# python3 -u ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --plotting --alignQuery="all" --verbose --saveResults