#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J s14c_temp_mahal_dist_rauc_2021

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s14c_temp_mahal_dist_rauc_2021.out
#SBATCH -e ../../batch_logs/covariance/s14c_temp_mahal_dist_rauc_2021.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_27 201902_03 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202101_27)
for A in "${exps[@]}"
do
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  #
  ITERATOR="ma"
  ESTIMATOR='mahal_ledoit'
  #
  SELECTIONLIST="laplace_spectral_baseline, laplace_baseline, laplace_spectral_scaled_mahal_ledoit, laplace_scaled_mahal_ledoit"
  # exp202101211100  exp202101271100
  python -u "./plotSignalRecruitmentAcrossExp.py" --expList="exp202101211100, exp202101271100" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
done
