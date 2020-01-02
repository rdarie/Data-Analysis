#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignFull.stdout
#SBATCH -e ../batch_logs/%j-alignFull.errout

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

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./assembleExperimentData.py --exp=$EXP --processAsigs --processRasters
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS $WINDOW --chanQuery="fr_sqrt" --blockName="fr_sqrt"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS $WINDOW --chanQuery="rig" --blockName="rig"
python3 ./calcAlignedAsigs.py --exp=$EXP --processAll $LAZINESS $WINDOW --chanQuery="fr" --blockName="fr"
python3 ./calcAlignedRasters.py --exp=$EXP --processAll $LAZINESS $WINDOW --chanQuery="raster" --blockName="raster"
# qa
python3 ./calcUnitMeanFR.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./calcUnitCorrelation.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose --plotting
python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP --processAll --verbose
python3 ./calcTrialOutliers.py --exp=$EXP --processAll --selector=$SELECTOR --saveResults --plotting --inputBlockName="fr" --alignQuery="all" --unitQuery="fr" --verbose