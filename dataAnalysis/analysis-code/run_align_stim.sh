#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20190121

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignStim_20190127.stdout
#SBATCH -e ../batch_logs/%j-alignStim_20190127.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
EXP="exp201901271000"
LAZINESS="--lazy"
WINDOW="--window=long"
TRIALSELECTOR="--trialIdx=5"
# TRIALSELECTOR="--processAll"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --chanQuery="fr" --blockName="fr"  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --chanQuery="rig" --blockName="rig"  --alignFolderName=stim
#  python3 ./calcAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --chanQuery="fr_sqrt" --blockName="fr_sqrt"  --alignFolderName=stim
python3 ./calcAlignedRasters.py --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --chanQuery="raster" --blockName="raster"  --alignFolderName=stim
# qa
python3 ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="stimOn" --unitQuery="fr" --verbose --plotting --alignFolderName=stim
