#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignFull_20190121

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignFull_20190121.stdout
#SBATCH -e ../batch_logs/%j-alignFull_20190121.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901070700"
# EXP="exp201901201200"
EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
LAZINESS="--lazy"
WINDOW="--window=long"
# TRIALSELECTOR="--trialIdx=2"
TRIALSELECTOR="--processAll"
UNITSELECTOR="--selector=_minfrmaxcorr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./calcTrialOutliers.py --exp=$EXP $TRIALSELECTOR $UNITSELECTOR --plotting --inputBlockName="fr" --alignQuery="all" --unitQuery="fr" --verbose