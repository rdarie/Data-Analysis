#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J alignTemp

# Specify an output file
#SBATCH -o ../batch_logs/%j-alignTemp.stdout
#SBATCH -e ../batch_logs/%j-alignTemp.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201812051000"
# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
EXP="exp201901271000"
# SELECTOR="_minfr"
SELECTOR="_minfrmaxcorr"
# TRIALSELECTOR=--trialIdx=1
TRIALSELECTOR=--processAll

LAZINESS="--lazy"
# TRIALIDX="2"
# GLMBLOCKNAME="tdrCmbGLM"
# OLSBLOCKNAME="tdrAcr"
# GLMESTIMATOR="Trial00${TRIALIDX}_${GLMBLOCKNAME}_long_midPeak"
# OLSESTIMATOR="Trial00${TRIALIDX}_${OLSBLOCKNAME}_long_midPeak"
#
# SELECTOR="Trial00${TRIALIDX}_minfrmaxcorr"
#
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

python3 ./plotClosestImpedanceMap.py --exp=$EXP $TRIALSELECTOR --verbose
