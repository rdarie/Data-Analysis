#!/bin/bash

# 08: Calculate binarized array and relevant analogsignals
# Request 24 hours of runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J analysis_calc_20200911

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-analysis_calc_20200911.stdout
#SBATCH -e ../../batch_logs/%j-%a-analysis_calc_20200911.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3,4

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901271000"
EXP="exp202009111100"
# EXP="exp201805231100"
LAZINESS="--lazy"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

SLURM_ARRAY_TASK_ID=1
# python3 -u ./synchronizeSIMItoNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP
python3 -u ./calcProprioAnalysisNix.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --chanQuery="notainp" --verbose --rigOnly
python3 -u ./calcMotionAlignTimes.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --plotParamHistograms $LAZINESS
python3 -u ./calcFR.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID