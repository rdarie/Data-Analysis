#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J calc_qa

# Specify an output file
#SBATCH -o ../../batch_logs/%j-calc_qa.stdout
#SBATCH -e ../../batch_logs/%j-calc_qa.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
EXP="exp201901271000"

LAZINESS="--lazy"
TRIALSELECTOR="--blockIdx=5"
# TRIALSELECTOR="--processAll"

# ALIGNQUERY="--alignQuery=midPeak"
ALIGNQUERY="--alignQuery=stimOn"
ALIGNFOLDER="--alignFolderName=stim"
# WINDOW="--window=long"
WINDOW="--window=miniRC"
BLOCKSELECTOR="--inputBlockName=fr"
ANALYSISNAME="--analysisName=loRes"
# ANALYSISNAME="--analysisName=default"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# first pass
UNITSELECTOR=""
# python3 -u ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISNAME $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose
# python3 -u ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISNAME $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --plotting
# python3 -u ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR $ALIGNFOLDER $ANALYSISNAME $WINDOW $LAZINESS --verbose

# remove outlier trials
UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
python3 -u ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISNAME $ALIGNQUERY $LAZINESS $BLOCKSELECTOR --saveResults --plotting --verbose --unitQuery="fr" --amplitudeFieldName="amplitude" --sqrtTransform

# recalculate, once outliers do not affect the calculation
# python3 ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISNAME $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --maskOutlierBlocks
# python3 ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ANALYSISNAME $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --plotting --maskOutlierBlocks
# python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR $ALIGNFOLDER $ANALYSISNAME $WINDOW $LAZINESS --verbose