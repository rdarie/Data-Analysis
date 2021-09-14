#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J plot_dataframes_201901_27

# Specify an output file
#SBATCH -o ../../batch_logs/plot_dataframes_201901_27.out
#SBATCH -e ../../batch_logs/plot_dataframes_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2


#   SBATCH --ntasks=4
#   SBATCH --ntasks-per-core=4
#   SBATCH --mem-per-cpu=64G

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_27.sh
source shellScripts/calc_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# PAGELIMITS="--limitPages=8"
OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
#
# iterators=(ma)
# targets=(lfp_CAR_mahal_ledoit lfp_CAR_spectral_mahal_ledoit)
# for ITER in "${iterators[@]}"
# do
#   for TARGET in "${targets[@]}"
#   do
#     python -u './plotSignalDataFrame.py' --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET $OPTS --verbose=1
#   done
# done

iterators=(pa)
for ITER in "${iterators[@]}"
do
  # python -u './plotMiscAdditionalFigures.py' --plotSuffix="rig_illustration" --verbose=1 --selectionName=rig $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  # python -u './plotSignalDataFrameHistogram.py' --plotSuffix="rig_illustration" --verbose=1 --selectionName=rig $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  # python -u './plotSignalDataFrameHistogram.py' --plotSuffix="spectral_illustration" --verbose=1 --selectionName=lfp_CAR_spectral $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  #
  # python -u './plotSignalDataFrame.py' --plotSuffix="rig_illustration" --verbose=1 --selectionName=rig $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  python -u './plotSignalDataFrame.py' --plotSuffix="lfp_illustration" --verbose=1 --selectionName=lfp_CAR $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  # python -u './plotSignalDataFrame.py' --plotSuffix="spectral_illustration" --verbose=1 --selectionName=lfp_CAR_spectral $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  #
  # python -u './plotSignalDataFrame.py' --plotSuffix="spectral_illustration" --verbose=1 --selectionName=lfp_CAR_spectral_scaled $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  # python -u './plotSignalDataFrame.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName=lfp_CAR_spectral_scaled_mahal_ledoit $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
  # python -u './plotSignalDataFrame.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName=lfp_CAR_mahal_ledoit $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
done

# ITER=ra
# python -u './plotSignalDataFrame.py' --plotSuffix="factor_illustration" --verbose=1 --selectionName=lfp_CAR_fa $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITER}"
