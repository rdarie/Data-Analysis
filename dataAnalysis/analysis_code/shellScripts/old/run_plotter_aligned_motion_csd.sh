#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_motion_csd

# Specify an output file
#SBATCH -o ../../batch_logs/%j_%a-plots_motion_csd.out
#SBATCH -e ../../batch_logs/%j_%a-plots_motion_csd.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2,3
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

# python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="csd" --unitQuery="lfp" --amplitudeFieldName=amplitude --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz" $HUEOPTS $OUTLIERMASK
python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --amplitudeFieldName=amplitude --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz, pedalMovementCat, pedalDirection, pedalSizeCat, freqBandName" $HUEOPTS $OUTLIERMASK
# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="csd" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="csd_spectral" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS