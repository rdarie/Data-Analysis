#!/bin/bash

# Request runtime:
#SBATCH --time=32:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J aligned_sigs

# Specify an output file
#SBATCH -o ../batch_logs/%j_aligned_sigs.stdout
#SBATCH -e ../batch_logs/%j_aligned_sigs.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcAsigsAlignedToStim.py' --exp=exp201901211000_alt --processAll
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcRasterAlignedToStim.py' --exp=exp201901211000_alt --processAll
