#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J calcAlignMiniRC

# Specify an output file
#SBATCH -o ../batch_logs/%j-calcAlignMiniRC.stdout
#SBATCH -e ../batch_logs/%j-calcAlignMiniRC.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

EXP="exp201901201200_alt"
# EXP="exp201901211000_alt"
# EXP="exp201901271000_alt"
MINIRCIDX="1"

#  python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcStimAlignTimes.py' --trialIdx=$MINIRCIDX --exp=$EXP
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcFR.py' --trialIdx=$MINIRCIDX --exp=$EXP
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcFRsqrt.py' --trialIdx=$MINIRCIDX --exp=$EXP
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcAsigsAlignedToStim.py' --trialIdx=$MINIRCIDX --exp=$EXP
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcRasterAlignedToStim.py' --trialIdx=$MINIRCIDX --exp=$EXP
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcAsigsAlignedToStim.py' --trialIdx=$MINIRCIDX --exp=$EXP --processShort
python3 '/gpfs/data/dborton/rdarie/Murdoc Neural Recordings/analysis-code/calcRasterAlignedToStim.py' --trialIdx=$MINIRCIDX --exp=$EXP --processShort
