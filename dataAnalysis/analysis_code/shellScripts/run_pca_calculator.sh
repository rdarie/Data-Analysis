#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J calc_pca

# Specify an output file
#SBATCH -o ../../batch_logs/%j-calc_pca.out
#SBATCH -e ../../batch_logs/%j-calc_pca.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901070700"
# EXP="exp201901201200"
EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
LAZINESS="--lazy"

SELECTOR="201901211000-Proprio_minfrmaxcorr"
# SELECTOR="201901271000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

#  python3 ./calcUnitMeanFR.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
#  python3 ./calcUnitCorrelation.py --exp=$EXP --processAll --inputBlockName="fr" --alignQuery="midPeak" --unitQuery="fr" --verbose
python3 ./selectUnitsByMeanFRandCorrelation.py --exp=$EXP --processAll --verbose
#  python3 ./calcPCAinChunks.py --exp=$EXP --processAll --window=long --selector=$SELECTOR $LAZINESS --verbose