#!/bin/bash

# 06a: Preprocess the NS5 File
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (32GB):
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J s00_preproc_dual_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/s00_preproc_dual_202101_27-%a.out
#SBATCH -e ../../batch_logs/s00_preproc_dual_202101_27-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=1-3

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# SLURM_ARRAY_TASK_ID=1

# exps=(201901_25 201901_26 201901_27 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
# exps=(201902_03 201902_04 201902_05)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 00 preprocess, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  
  ########### get analog inputs separately to run synchronization, etc
  # !! --maskMotorEncoder ignores all motor events outside alignTimeBounds
  # python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analogOnly
  python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --analogOnly --maskMotorEncoder
  
  ######### finalize dataset
  # # python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullUnfiltered --chunkSize=700
  ############################
  # old
  # python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=utah --fullSubtractMeanUnfiltered --chunkSize=700
  # python -u ./preprocNS5.py --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --arrayName=nform --fullSubtractMeanUnfiltered
  
  ########### get dataset to run spike extraction on
  # python -u ./preprocNS5.py --arrayName=utah --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean --chunkSize=700
  # python -u ./preprocNS5.py --arrayName=nform --exp=$EXP --blockIdx=$SLURM_ARRAY_TASK_ID --fullSubtractMean
  
  # python -u ./synchronizeNFormToNSP.py --blockIdx=$SLURM_ARRAY_TASK_ID --exp=$EXP --trigRate=100
done