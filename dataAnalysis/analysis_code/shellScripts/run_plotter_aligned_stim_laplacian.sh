#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J plots_stim_lapl

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-plots_stim_lapl.out
#SBATCH -e ../../batch_logs/%j-%a-plots_stim_lapl.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=1,2

# SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_plotter_aligned_stim_preamble.sh

python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="kcsd" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="kcsd" --unitQuery="lfp" --amplitudeFieldName=amplitude --exp=$EXP $WINDOW $TIMEWINDOWOPTS $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz, pedalMovementCat" $HUEOPTS $OUTLIERMASK

# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="kcsd_pca_spectral" --unitQuery="pca" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS