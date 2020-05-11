#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=128G

# Specify a job name:
#SBATCH -J plotsLDA

# Specify an output file
#SBATCH -o ../../batch_logs/%j-plotsLDA.stdout
#SBATCH -e ../../batch_logs/%j-plotsLDA.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotPCAalignedToStimNCM2019LDA.py'
