#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=100:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_previews_202101_28

# Specify an output file
#SBATCH -o ../../batch_logs/preproc_previews_202101_28-%a.out
#SBATCH -e ../../batch_logs/preproc_previews_202101_28-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1-3
#SBATCH --export=CCV_HEADLESS=1

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# EXP="exp201901070700"

# EXP="exp201901221000"
# has 1 minirc 2-3 motion
# EXP="exp201901231000"
# has 1 motion
# EXP="exp201901240900"
# has 1 minirc 2 motion
# EXP="exp201901251000"
# has 1 minirc 2 motion
# EXP="exp201901261000"
# has 1-3 motion 4 minirc
# EXP="exp201901271000"
# has 1-4 motion 5 minirc
# EXP="exp201901281200"
# has 1-4 motion
# EXP="exp201901291000"
# has 1-3 motion
# EXP="exp201901301000"
# has 1-3 motion 4 minirc
# EXP="exp201901311000"
# has 1-4 motion 5 minirc
EXP="exp201902010900"
#  has 1-4 motion 5 minirc
EXP="exp201902021100"
# has 3-5 motion 6 minirc; blocks 1 and 2 were bad;
# EXP="exp201902031100"
# has 1-4 motion 5 minirc;
# EXP="exp201902041100"
# has 1-4 motion 5 minirc;
# EXP="exp201902051100"
# has 1-4 motion
# EXP="exp202101111100"
# EXP="exp202101141100"
# EXP="exp202101061100"
# EXP="exp202101191100"
EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
EXP="exp202101281100"
# EXP="exp202102021100"
# EXP="exp202102031100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"

python -u ./previewINSSessionSummary.py --exp=$EXP --reprocessAll
# python -u ./saveImpedances.py --exp=$EXP --processAll --reprocess

# for BLOCKIDX in 1 2
# do
#     python -u './previewNSPTapTimes.py' --blockIdx=$BLOCKIDX --exp=$EXP
# done
#  SLURM_ARRAY_TASK_ID=1
# python -u './previewNSPTapTimes.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
