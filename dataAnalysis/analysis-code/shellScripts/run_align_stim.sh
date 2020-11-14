#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20201027

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignStim_20201027.stdout
#SBATCH -e ../../batch_logs/%j-alignStim_20201027.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202003091200"
EXP="exp202010271200"

LAZINESS="--lazy"
WINDOW="--window=XS"
# TRIALSELECTOR="--blockIdx=5"
TRIALSELECTOR="--processAll"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# python -u ./assembleExperimentData.py --exp=$EXP $ANALYSISFOLDER --processAsigs --processRasters

python -u ./calcAlignedAsigs.py --chanQuery="lfp" --outputBlockName="lfp" --eventBlockName='analyze' --signalBlockName='analyze' --verbose --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNSELECTOR --eventName=stimAlignTimes --alignFolderName=stim
# python -u ./calcAlignedAsigs.py --chanQuery="fr" --outputBlockName="fr" --eventBlockName='analyze' --signalBlockName='fr' --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --alignFolderName=stim
python -u ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockName="rig" --eventBlockName='analyze' --signalBlockName='analyze' --verbose --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --alignFolderName=stim
python -u ./calcAlignedAsigs.py --chanQuery="raster" --outputBlockName="raster" --eventBlockName='analyze' --signalBlockName='binarized' --verbose --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --alignFolderName=stim
# qa
# python ./calcBlockOutliers.py --exp=$EXP $TRIALSELECTOR --inputBlockName="fr" --alignQuery="stimOn" --unitQuery="fr" --verbose --plotting --alignFolderName=stim
