#!/bin/bash
# 09: Assemble binarized array and relevant analogsignals
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J trial_assembly

# Specify an output file
#SBATCH -o ../batch_logs/%j-trial_assembly.stdout
#SBATCH -e ../batch_logs/%j-trial_assembly.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201812051000"
python3 './assembleExperimentData.py' --exp=$EXP --processAsigs