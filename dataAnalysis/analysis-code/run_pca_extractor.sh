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
#SBATCH -o ../batch_logs/%j-extract_pca.stdout
#SBATCH -e ../batch_logs/%j-extract_pca.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"
# ESTIMATOR="201901211000-Proprio_pca"
ESTIMATOR="201901271000-Proprio_pca"
# ESTIMATOR="201901201200-Proprio_pca"

#  python3 './applyEstimatorToAsig.py' --exp=$EXP --processAll --estimator=$ESTIMATOR
python3 './applyEstimatorToTriggered.py' --exp=$EXP --processAll --estimator=$ESTIMATOR
