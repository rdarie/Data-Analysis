#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=96G

# Specify a job name:
#SBATCH -J s04e_prelim_plots_rd_201901_27

# Specify an output file
#SBATCH -o ../../batch_logs/s04e_prelim_plots_rd_201901_27.out
#SBATCH -e ../../batch_logs/s04e_prelim_plots_rd_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

#    SLURM_ARRAY_TASK_ID=2
#
#  202101_21 202101_22
# exps=(201901_25 201901_26 201901_27 201902 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(201901_27)
for A in "${exps[@]}"
do
  echo "step 04e plots, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  ITERATOR="rc"
  #
  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="rig_illustration" --verbose=1 --selectionName="rig" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="laplace_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="laplace_spectral_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  # ##
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_topo_illustration" --verbose=1 --selectionName="laplace_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration_topo" --verbose=1 --selectionName="laplace_spectral_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
done