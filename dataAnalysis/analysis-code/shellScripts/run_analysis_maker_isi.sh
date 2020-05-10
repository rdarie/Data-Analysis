#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=200G

# Specify a job name:
#SBATCH -J analysis_mini_20200427

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-analysis_mini_20200427.stdout
#SBATCH -e ../batch_logs/%j-%a-analysis_mini_20200427.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=3

# EXP="exp202003091200"
# EXP="exp202003181300"
# EXP="exp202003191400"
# EXP="exp202003201200"
# EXP="exp202004251400"

# EXP="exp202004271200"
# has blocks 1-4

# EXP="exp202004301200"
# has blocks 4,5

EXP="exp202005011400"
# has blocks 1,2,3,4,5,6

# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"

# WINDOW="--window=short"
# WINDOW="--window=miniRC"
WINDOW="--window=extraShort"
# WINDOW="--window=extraExtraShort"

# TRIALSELECTOR="--processAll"
# TRIALSELECTOR="--blockIdx=2"

# ANALYSISSELECTOR="--analysisName=emg1msec"
# ANALYSISSELECTOR="--analysisName=emg1msecSmooth"
# ANALYSISSELECTOR="--analysisName=emg1msecNoLFPFilterSmoothEMG"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"
# ANALYSISSELECTOR="--analysisName=emgStretchTime"
# ANALYSISSELECTOR="--analysisName=emgHiRes"
ANALYSISSELECTOR="--analysisName=emgLoRes"

# CHANSELECTOR="--chanQuery=all"
# CHANSELECTOR="--chanQuery=isiemgraw"
CHANSELECTOR="--chanQuery=isiemg"
# CHANSELECTOR="--chanQuery=isispinal"
# CHANSELECTOR="--chanQuery=isispinaloremg"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

# SLURM_ARRAY_TASK_ID=1
python -u './calcISIAnalysisNix.py' --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID $CHANSELECTOR $ANALYSISSELECTOR