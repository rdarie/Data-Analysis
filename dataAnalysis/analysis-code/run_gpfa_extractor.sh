#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=59G

# Specify a job name:
#SBATCH -J extract_GPFA

# Specify an output file
#SBATCH -o ../batch_logs/%j-extract_GPFA.stdout
#SBATCH -e ../batch_logs/%j-extract_GPFA.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# EXP="exp201901201200"
# SELECTOR="201901211000-Proprio_minfr"
SELECTOR="201901271000-Proprio_minfrmaxcorr"
# SELECTOR="201901201200-Proprio_minfr"

python3 './applyGPFAtoTriggered.py' --exp=$EXP --window=long --processAll --selector=$SELECTOR --verbose
