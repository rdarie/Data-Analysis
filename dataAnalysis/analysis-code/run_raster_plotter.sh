#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J plotRaster

# Specify an output file
#SBATCH -o ../batch_logs/plotRaster-o-%j.out
#SBATCH -e ../batch_logs/plotRaster-e-%j.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotTwinAlignedToStim.py'
