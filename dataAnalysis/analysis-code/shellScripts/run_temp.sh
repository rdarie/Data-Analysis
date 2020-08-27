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
# EXP="exp201901271000"
EXP="exp201901261000"
# EXP="exp201901070700"

# ALIGNFOLDER="--analysisName=loRes"
ALIGNFOLDER="--analysisName=default"

SLURM_ARRAY_TASK_ID=4
TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# TRIALSELECTOR="--processAll"

# UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
UNITSELECTOR=""

LAZINESS="--lazy"

# WINDOW="--window=miniRC"
WINDOW="--window=XS"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python3 ./calcAlignedAsigs.py --chanQuery="utahlfp" --outputBlockName="utahlfp"     --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNFOLDER --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --chanQuery="fr" --outputBlockName="fr"             --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNFOLDER --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedAsigs.py --chanQuery="rig" --outputBlockName="rig"           --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNFOLDER --eventName=stimAlignTimes  --alignFolderName=stim
# python3 ./calcAlignedAsigs.py --chanQuery="fr_sqrt" --outputBlockName="fr_sqrt"   --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNFOLDER --eventName=stimAlignTimes  --alignFolderName=stim
python3 ./calcAlignedRasters.py --chanQuery="raster" --outputBlockName="raster"   --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $ALIGNFOLDER --eventName=stimAlignTimes  --alignFolderName=stim
#
python3 -u ./calcUnitMeanFR.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose
python3 -u ./calcUnitCorrelation.py --exp=$EXP $TRIALSELECTOR $WINDOW $ALIGNFOLDER $ALIGNQUERY --inputBlockName="fr" --unitQuery="fr" --verbose --plotting
python3 -u ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP $TRIALSELECTOW $ALIGNFOLDER $WINDOW $LAZINESS --verbose

UNITSELECTOR="--selector=unitSelector_minfrmaxcorr"
python3 -u ./plotAlignedNeurons.py --exp=$EXP $TRIALSELECTOR $ALIGNFOLDER $UNITSELECTOR $WINDOW --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
python3 -u ./plotAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $ALIGNFOLDER $WINDOW --inputBlockName="rig" --unitQuery="all" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides
python3 -u ./plotAlignedAsigs.py --exp=$EXP $TRIALSELECTOR $ALIGNFOLDER $WINDOW --inputBlockName="utahlfp" --unitQuery="utahlfp" --alignQuery="stimOn" --rowName="RateInHz" --hueName="amplitude" --alignFolderName=stim --enableOverrides