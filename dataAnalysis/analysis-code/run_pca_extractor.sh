#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J extract_PCA

# Specify an output file
#SBATCH -o ../batch_logs/%j-extract_PCA.stdout
#SBATCH -e ../batch_logs/%j-extract_PCA.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp201901211000_alt"
# EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"
ESTIMATOR="201901211000-Proprio_pca"
# ESTIMATOR="201901271000-Proprio_pca"
# ESTIMATOR="201901201200-Proprio_pca"

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/applyEstimatorToAsig.py' --exp=$EXP --processAll --estimator=$ESTIMATOR
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/applyEstimatorToTriggered.py' --exp=$EXP --processAll --estimator=$ESTIMATOR
