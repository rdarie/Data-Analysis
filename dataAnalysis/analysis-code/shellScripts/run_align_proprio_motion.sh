#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull_rig

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignFull_rig.stdout
#SBATCH -e ../../batch_logs/%j-alignFull_rig.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# EXP="exp201805231100"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
# EXP="exp202009111100"
# EXP="exp202009211200"
EXP="exp202010011100"

LAZINESS="--lazy"

# WINDOW="--window=XS"
# WINDOW="--window=short"
WINDOW="--window=miniRC"

# SLURM_ARRAY_TASK_ID=1
# BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
BLOCKSELECTOR="--processAll"

# UNITSELECTOR="--selector=_minfrmaxcorr"
UNITSELECTOR=""

ALIGNSELECTOR="--eventName=motionAlignTimes"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"
OUTLIERSWITCH=""
# OUTLIERSWITCH="--maskOutlierBlocks"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# python3 -u ./assembleExperimentData.py --exp=$EXP --processRasters

python3 -u ./calcAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $LAZINESS $WINDOW $ALIGNSELECTOR $ANALYSISFOLDER --chanQuery="rig" --outputBlockName="rig" --verbose

# python3 -u ./calcAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $LAZINESS $WINDOW $ALIGNSELECTOR $ANALYSISFOLDER --chanQuery="fr" --outputBlockName="fr" --verbose
# python3 -u ./calcAlignedRasters.py --exp=$EXP $BLOCKSELECTOR $LAZINESS $WINDOW $ALIGNSELECTOR $ANALYSISFOLDER --chanQuery="raster" --outputBlockName="raster" --verbose

# python3 -u ./calcAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $LAZINESS $WINDOW $ALIGNSELECTOR $ANALYSISFOLDER --chanQuery="lfp" --outputBlockName="lfp" --verbose

ALIGNQUERY="--alignQuery=outbound"
python3 -u ./plotAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $WINDOW --inputBlockName="rig" --unitQuery="rig" $ALIGNQUERY --rowName="pedalSizeCat" --colControl= --colName="pedalDirection" --rowControl= $OUTLIERSWITCH --verbose --enableOverrides

# python3 -u ./plotAlignedNeurons.py --exp=$EXP $BLOCKSELECTOR $WINDOW $ALIGNQUERY $UNITSELECTOR --rowName="pedalSizeCat" --rowControl= --colControl= --colName="pedalDirection" $OUTLIERSWITCH --verbose --enableOverrides

# python3 -u ./plotAlignedAsigs.py --exp=$EXP $BLOCKSELECTOR $WINDOW --inputBlockName="lfp" --unitQuery="lfp" $ALIGNQUERY --rowName= --rowControl= --colControl= --colName="pedalDirection" $OUTLIERSWITCH --verbose --enableOverrides