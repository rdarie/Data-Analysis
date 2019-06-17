#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J calcAlignFull

# Specify an output file
#SBATCH -o ../batch_logs/%j-calcAlignFull.stdout
#SBATCH -e ../batch_logs/%j-calcAlignFull.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000_alt"
EXP="exp201901271000_alt"

#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/assembleExperimentData.py' --exp=$EXP
#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcMotionStimAlignTimes.py' --trialIdx=1 --exp=$EXP --processAll --plotParamHistograms
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcFR.py' --trialIdx=1 --exp=$EXP --processAll
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcFRsqrt.py' --trialIdx=1 --exp=$EXP --processAll
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcAsigsAlignedToMotion.py' --exp=$EXP --processAll --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotAsigsAlignedToMotionStim.py' --exp=$EXP --processAll --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcRasterAlignedToMotion.py' --exp=$EXP --processAll --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/plotNeuronsAlignedToMotionStim.py' --exp=$EXP --processAll --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcAsigsAlignedToMotion.py' --exp=$EXP --processAll
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcRasterAlignedToMotion.py' --exp=$EXP --processAll
