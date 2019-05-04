#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J plotsMinIRC

# Specify an output file
#SBATCH -o ../batch_logs/%j-plotsMinIRC.stdout
#SBATCH -e ../batch_logs/%j-plotsMinIRC.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotMiniRCalignedToStimNCM2019.py'
