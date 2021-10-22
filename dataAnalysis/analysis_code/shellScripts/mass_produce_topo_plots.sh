#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J mass_topo_plots_2021

# Specify an output file
#SBATCH -o ../../batch_logs/mass_topo_plots_2021.out
#SBATCH -e ../../batch_logs/mass_topo_plots_2021.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=2



exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
for A in "${exps[@]}"
do
  echo "plotting $A"
  SLURM_ARRAY_TASK_ID=2
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_preamble.sh
  BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
  TIMEWINDOWOPTS="--winStart=-600 --winStop=75"
  PAGELIMITS=""
  HUEOPTS="--hueName=trialAmplitude --hueControl="
  OUTLIERMASK="--maskOutlierBlocks"
  python -u ./plotAlignedAsigsTopo.py --inputBlockSuffix="lfp" --unitQuery="lfp" --alignQuery="stimOnHighRate" --alignFolderName=stim $TIMEWINDOWOPTS $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, trialRateInHz" $HUEOPTS $OUTLIERMASK --exp=$EXP $WINDOW $ANALYSISFOLDER $BLOCKSELECTOR
done
