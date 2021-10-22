#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=125G

# Specify a job name:
#SBATCH -J plot_df_202101_20

# Specify an output file
#SBATCH -o ../../batch_logs/plot_dataframes_02101_20.out
#SBATCH -e ../../batch_logs/plot_dataframes_02101_20.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999


#   SBATCH --ntasks=4
#   SBATCH --ntasks-per-core=4
#   SBATCH --mem-per-cpu=64G

# SLURM_ARRAY_TASK_ID=2
exps=(202101_20 202101_21 202101_22)
for A in "${exps[@]}"
do
  echo "step 03 assemble plot normalize, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source shellScripts/calc_aligned_motion_preamble.sh
  # source shellScripts/calc_aligned_stim_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  # PAGELIMITS="--limitPages=8"
  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  ITER="pa"
  python -u './plotSignalDataFrame.py' --plotSuffix="lfp_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  ITER="ma"
  python -u './plotSignalDataFrame.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="laplace_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  python -u './plotSignalDataFrame.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="laplace_spectral_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  python -u './plotSignalDataFrame.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="laplace_spectral_scaled_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  python -u './plotSignalDataFrame.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="laplace_scaled_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  #
  python -u './plotFinalSynchConfirmation.py' --plotSuffix="final_synch_confirmation" --winStart="-250" --winStop="750" --verbose=1 --selectionName="laplace_scaled" --selectionName2="laplace_spectral_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
done