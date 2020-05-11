#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull_20190127

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignFull_20190127.stdout
#SBATCH -e ../batch_logs/%j-alignFull_20190127.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
EXP="exp201901271000"
LAZINESS="--lazy"
WINDOW="--window=long"
# TRIALSELECTOR="--blockIdx=2"
TRIALSELECTOR="--processAll"
UNITSELECTOR="--selector=_minfrmaxcorr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
# python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW --chanQuery="fr_sqrt" --blockName="fr_sqrt"
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW --chanQuery="rig" --blockName="rig" --verbose
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW --chanQuery="fr" --blockName="fr" --verbose
python3 ./calcAlignedRasters.py --exp=$EXP $TRIALSELECTOR $LAZINESS $WINDOW --chanQuery="raster" --blockName="raster" --verbose
# qa
python3 ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting

# python3 ./selectUnitsByMeanFRandCorrelationAndAmplitude.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --verbose
# or
# python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --verbose

# python3 ./calcBlockOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --plotting --alignQuery="all" --verbose --saveResults