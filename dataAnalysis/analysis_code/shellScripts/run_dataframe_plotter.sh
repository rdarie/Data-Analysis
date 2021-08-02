#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=4
#SBATCH --mem=120G

# Specify a job name:
#SBATCH -J plot_dataframes_27

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-plot_dataframes_27.out
#SBATCH -e ../../batch_logs/%j-%a-plot_dataframes_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2


#   SBATCH --ntasks=4
#   SBATCH --ntasks-per-core=4
#   SBATCH --mem-per-cpu=64G

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble.sh
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# PAGELIMITS="--limitPages=8"
OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER} ${TIMEWINDOWOPTS} ${STATSOVERLAY} ${HUEOPTS} ${ROWOPTS} ${COLOPTS} ${STYLEOPTS} ${SIZEOPTS} ${PAGELIMITS} ${OTHERASIGOPTS}"
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
iterators=(rd)
targets=(rig)
for ITER in "${iterators[@]}"
do
  for TARGET in "${targets[@]}"
  do
    python -u './plotSignalDataFrame.py' --datasetName="Block_${WINDOWTERM}_df_${ITER}" --selectionName=$TARGET $OPTS --verbose=1
  done
done