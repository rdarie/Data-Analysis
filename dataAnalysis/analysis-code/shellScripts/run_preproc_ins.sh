#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J ins_preproc_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_preproc_20190127.stdout
#SBATCH -e ../../batch_logs/%j-%a-ins_preproc_20190127.errout

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3,4

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# EXP="exp201901070700"
# EXP="exp201901201200"
# EXP="exp201901211000"
# EXP="exp201901221000"
# EXP="exp201901231000"
# EXP="exp201901261000"
EXP="exp201901271000"

# SLURM_ARRAY_TASK_ID=5

# --makePlots to make quality check plots
# --showPlots to interactively display quality check plots
# --disableStimDetection to use HUT derived stim start times
# python3 -u './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --makePlots --verbose
python3 -u './preprocINS.py' --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --makePlots --verbose |& tee "../../batch_logs/${EXP}_Block_${SLURM_ARRAY_TASK_ID}_preproc_ins"
