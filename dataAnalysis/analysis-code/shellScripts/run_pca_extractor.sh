#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J extract_pca

# Specify an output file
#SBATCH -o ../../batch_logs/%j-extract_pca.stdout
#SBATCH -e ../../batch_logs/%j-extract_pca.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# EXP="exp201901201200"
# ESTIMATOR="201901211000-Proprio_pca_long_midPeak"
# ESTIMATOR="201901271000-Proprio_pca_long_midPeak"
ESTIMATOR="201901271000-Proprio_tdrAcr_long_midPeak"

#  python3 './applyEstimatorToAsig.py' --exp=$EXP --processAll --estimator=$ESTIMATOR
python3 './applyEstimatorToTriggered.py' --exp=$EXP --processAll --window="long" --alignQuery="midPeak" --estimator=$ESTIMATOR --lazy --profile --verbose
