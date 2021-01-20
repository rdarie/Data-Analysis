#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J ins_preproc

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_preproc.stdout
#SBATCH -e ../../batch_logs/%j-%a-ins_preproc.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202006171300"
# EXP="exp202007011300"
# EXP="exp202007021300"
# EXP="exp202007071300"
# EXP="exp202007081300"
# EXP="exp202010011100"
# EXP="exp202009231400"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202012171200"
EXP="exp202101061100"
EXP="exp202101111100"
EXP="exp202101141100"

# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# global operations
python './shuttleFilesToFromScratch.py' --exp=$EXP --everything --fromScratchToData
# 
