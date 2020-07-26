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
EXP="exp201901271000"

# 

LAZINESS="--lazy"
# TRIALSELECTOR=--blockIdx=2
TRIALSELECTOR="--processAll"

ALIGNQUERY="--alignQuery=midPeak"
WINDOW="--window=long"
BLOCKSELECTOR="--inputBlockName=fr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# first pass
UNITSELECTOR=""
python3 ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting
python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR --verbose

# remove outlier trials
UNITSELECTOR="--selector=_minfrmaxcorr"
python3 ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR $WINDOW $LAZINESS $BLOCKSELECTOR --saveResults --plotting $ALIGNQUERY --verbose --alignFolderName=motion --unitQuery="fr" --amplitudeFieldName="amplitude" --sqrtTransform

# recalculate, once outliers do not affect the calculation
python3 ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --maskOutlierBlocks
python3 ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting --maskOutlierBlocks
python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR --verbose
