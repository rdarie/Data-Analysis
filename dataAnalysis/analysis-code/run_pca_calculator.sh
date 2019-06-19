#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J calc_pca

# Specify an output file
#SBATCH -o ../batch_logs/%j-calc_pca.stdout
#SBATCH -e ../batch_logs/%j-calc_pca.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"
# EXP="exp201901201200_alt"

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcPCA.py' --exp=$EXP --processAll
