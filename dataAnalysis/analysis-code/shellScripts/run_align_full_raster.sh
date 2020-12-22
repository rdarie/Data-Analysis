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
# EXP="exp201901271000"
# EXP="exp202010011100"
EXP="exp202012121100"

LAZINESS="--lazy"

# WINDOW="--window=long"
WINDOW="--window=M"

# ANALYSISFOLDER="--analysisName=loRes"
ANALYSISFOLDER="--analysisName=default"

SLURM_ARRAY_TASK_ID=1
TRIALSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID}"
# TRIALSELECTOR="--processAll"

EVENTSELECTOR="--eventName=motionAlignTimes"
ALIGNFOLDER="--alignFolderName=motion"

python -u ./calcAlignedAsigs.py --chanQuery="raster" --outputBlockName="raster" --eventBlockName='analyze' --signalBlockName='binarized' --verbose --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS $EVENTSELECTOR $ALIGNFOLDER
