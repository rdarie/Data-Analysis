#!/bin/bash

# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=64G

# Specify a job name:
#SBATCH -J plotsMotionPlus

# Specify an output file
#SBATCH -o ../batch_logs/%j-plotsMotionPlus.stdout
#SBATCH -e ../batch_logs/%j-plotsMotionPlus.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp201901211000_alt"
# EXP="exp201901271000_alt"

#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotNeuronsAlignedToMotionStim.py' --exp=$EXP --processAll
#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotAsigsAlignedToMotionStim.py' --exp=$EXP --processAll
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotNeuronsAlignedToMotionStim.py' --exp=$EXP --processAll  --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotAsigsAlignedToMotionStim.py' --exp=$EXP --processAll  --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotNeuronsAlignedToMotionStim.py' --exp=$EXP --processAll  --processShort --alignQuery=\(pedalMovementCat=='return'\)
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotAsigsAlignedToMotionStim.py' --exp=$EXP --processAll  --processShort --alignQuery=\(pedalMovementCat=='return'\)
