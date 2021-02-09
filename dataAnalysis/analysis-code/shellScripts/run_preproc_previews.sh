#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=100:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_previews_2021_01_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-preproc_previews_2021_01_27.out
#SBATCH -e ../../batch_logs/%j-%a-preproc_previews_2021_01_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
##############SBATCH --array=1,2

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

EXP="exp202101141100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"

# python -u ./previewINSSessionSummary.py --exp=$EXP
python -u ./saveImpedances.py --exp=$EXP --processAll --reprocess

# for BLOCKIDX in 1 2 3
# do
#     python -u './previewNSPTapTimes.py' --blockIdx=$BLOCKIDX --exp=$EXP --usedTENSPulses
# done