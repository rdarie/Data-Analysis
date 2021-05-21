#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J plots_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j_%a-plots_motion_lfp.out
#SBATCH -e ../../batch_logs/%j_%a-plots_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

# PAGELIMITS="--limitPages=8"
# python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
# #
# python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
#
# python3 -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" --amplitudeFieldName=amplitude --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz, pedalMovementCat, pedalDirection, pedalSizeCat, freqBandName" $HUEOPTS $OUTLIERMASK
# python3 -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --amplitudeFieldName=amplitude --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR --groupPagesBy="electrode, RateInHz, pedalMovementCat, pedalDirection, pedalSizeCat" $HUEOPTS $OUTLIERMASK
#
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
# BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"
python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral_fa" --unitQuery="factor" --enableOverrides --exp=$EXP $BLOCKSELECTOR $ANALYSISFOLDER $WINDOW $ALIGNQUERY $ALIGNFOLDER $TIMEWINDOWOPTS $STATSOVERLAY $OUTLIERMASK $HUEOPTS $ROWOPTS $COLOPTS $STYLEOPTS $SIZEOPTS $PAGELIMITS $OTHERASIGOPTS
#