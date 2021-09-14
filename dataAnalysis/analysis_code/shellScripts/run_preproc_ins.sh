#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=100:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J preproc_ins_201901_26

# Specify an output file
#SBATCH -o ../../batch_logs/preproc_ins_201901_26-%a.out
#SBATCH -e ../../batch_logs/preproc_ins_201901_26-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3


module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

EXP="exp201901261000"
# EXP="exp201901271000"

# EXP="exp202101141100"
# EXP="exp202101061100"
# EXP="exp202101191100"
# EXP="exp202101201100"
# EXP="exp202101211100"
# EXP="exp202101221100"
# EXP="exp202101251100"
# EXP="exp202101271100"
# EXP="exp202101281100"
# EXP="exp202102041100"
# EXP="exp202102081100"
# EXP="exp202102101100"
# EXP="exp202102151100"

SLURM_ARRAY_TASK_ID=1

# --makePlots to make quality check plots
# --showPlots to interactively display quality check plots
# --disableStimDetection to use HUT derived stim start times

python -u './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --makePlots --verbose --disableStimDetection

# python -u './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --makePlots --verbose |& tee "../../batch_logs/${EXP}_Block_${SLURM_ARRAY_TASK_ID}_preproc_ins"
