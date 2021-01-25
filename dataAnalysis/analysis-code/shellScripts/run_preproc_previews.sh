#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=100:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_previews_2021_01_20

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-preproc_previews_2021_01_20.stdout
#SBATCH -e ../../batch_logs/%j-%a-preproc_previews_2021_01_20.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
##############SBATCH --array=1,2

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
# EXP="exp201901271000"
# EXP="exp202010201200"
# EXP="exp202010251400"
# EXP="exp202010261100"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011231200"
# EXP="exp202011271100"
# EXP="exp202011301200"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012151200"
# EXP="exp202012171200"
# EXP="exp202101051100"
# EXP="exp202101061100"
# EXP="exp202101111100"
# EXP="exp202101111100"
# EXP="exp202101141100"
EXP="exp202101191100"
EXP="exp202101201100"
EXP="exp202101211100"

python -u ./previewINSSessionSummary.py --exp=$EXP
SLURM_ARRAY_TASK_ID=1
python -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --usedTENSPulses
SLURM_ARRAY_TASK_ID=2
python -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --usedTENSPulses
SLURM_ARRAY_TASK_ID=3
python -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --usedTENSPulses
SLURM_ARRAY_TASK_ID=4
python -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --usedTENSPulses