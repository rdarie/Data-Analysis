#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=72:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J alignStim_20200318

# Specify an output file
#SBATCH -o ../../batch_logs/%j-alignStim_20200318.stdout
#SBATCH -e ../../batch_logs/%j-alignStim_20200318.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
EXP="exp202004271200"
# EXP="exp202003201200"

LAZINESS="--lazy"
# LAZINESS=""
# WINDOW="--window=miniRC"
WINDOW="--window=short"
# WINDOW="--window=extraShort"
# WINDOW="--window=extraExtraShort"
# TRIALSELECTOR="--blockIdx=2"
# TRIALSELECTOR="--blockIdx=3"
TRIALSELECTOR="--processAll"
# ANALYSISSELECTOR="--analysisName=emg"
# ANALYSISSELECTOR="--analysisName=default"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emg1msec"
# ANALYSISSELECTOR="--analysisName=emg1msecSmooth"
# ANALYSISSELECTOR="--analysisName=emg1msecNoLFPFilterSmoothEMG"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
#
# UNITSELECTOR="--unitQuery=all"
# UNITSELECTOR="--unitQuery=isiemgraw"
# UNITSELECTOR="--unitQuery=isispinal"
UNITSELECTOR="--unitQuery=isispinaloremg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

python3 ./launchVis.py