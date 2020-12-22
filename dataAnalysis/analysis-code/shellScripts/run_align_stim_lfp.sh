#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=16G

# Specify a job name:
#SBATCH -J lfp_alignStim_20201027

# Specify an output file
#SBATCH -o ../../batch_logs/%j-lfp_alignStim_20201027.stdout
#SBATCH -e ../../batch_logs/%j-lfp_alignStim_20201027.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202003091200"
# EXP="exp202010271200"
EXP="exp202011201100"
EXP="exp202012111100"
EXP="exp202012121100"

LAZINESS="--lazy"

# WINDOW="--window=XS"
WINDOW="--window=M"

TRIALSELECTOR="--blockIdx=1"
# TRIALSELECTOR="--processAll"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python -u ./calcAlignedAsigs.py --chanQuery="lfp" --outputBlockName="lfp" --eventBlockName='analyze' --signalBlockName='analyze' --verbose --exp=$EXP $TRIALSELECTOR $WINDOW $LAZINESS --eventName=stimAlignTimes --alignFolderName=stim
