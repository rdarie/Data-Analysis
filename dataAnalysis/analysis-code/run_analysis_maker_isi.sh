#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J analysis_mini_20190127

# Specify an output file
#SBATCH -o ../batch_logs/%j-%a-analysis_mini_20190127.stdout
#SBATCH -e ../batch_logs/%j-%a-analysis_mini_20190127.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=2,3,4

EXP="exp202003201200"
EXP="exp202003091200"
# SELECTOR="Block005_minfrmaxcorr"
SELECTOR="_minfrmaxcorr"
# WINDOW="--window=miniRC"
WINDOW="--window=extraShort"
# TRIALSELECTOR="--processAll"
TRIALSELECTOR="--blockIdx=2"
ANALYSISSELECTOR="--analysisName=emg"
ANALYSISSELECTOR="--analysisName=default"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

# SLURM_ARRAY_TASK_ID=2
python ./calcISIAnalysisNix.py --exp=$EXP $TRIALSELECTOR --chanQuery="all" $ANALYSISSELECTOR
# python ./calcStimAlignTimes.py --exp=$EXP $TRIALSELECTOR --plotParamHistograms
# python ./calcFR.py --exp=$EXP $TRIALSELECTOR
