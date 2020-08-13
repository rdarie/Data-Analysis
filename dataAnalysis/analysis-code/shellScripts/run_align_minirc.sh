#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignMiniRC

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignMiniRC.stdout
#SBATCH -e ../../batch_logs/%j-alignMiniRC.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# EXP="exp201901201200"
# EXP="exp201901221000"
EXP="exp201901271000"
# EXP="exp201901261000"
# EXP="exp201901070700"
TRIALSELECTOR="--blockIdx=4"
ALIGNSELECTOR="--analysisName=loRes"
# TRIALSELECTOR="--processAll"
# UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
UNITSELECTOR=""
LAZINESS="--lazy"
WINDOW="--window=miniRC"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python3 ./calcAlignedAsigs.py --chanQuery="utahlfp" --outputBlockName="utahlfp"     --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --chanQuery="fr" --outputBlockName="fr"             --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockName="rig"           --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes  --alignFolderName=stim
# python3 ./calcAlignedAsigs.py --chanQuery="fr_sqrt" --outputBlockName="fr_sqrt"   --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedRasters.py --chanQuery="raster" --outputBlockName="raster"   --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes  --alignFolderName=stim
