#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=128G

# Specify a job name:
#SBATCH -J plots_motion_lfp_27

# Specify an output file
#SBATCH -o ../../batch_logs/plots_motion_lfp_27.out
#SBATCH -e ../../batch_logs/plots_motion_lfp_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo

#SBATCH --export=CCV_HEADLESS=1
# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_exp_preamble_27.sh
source ./shellScripts/run_plotter_aligned_motion_preamble.sh

# PAGELIMITS="--limitPages=8"
OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNQUERY} ${ALIGNFOLDER} ${TIMEWINDOWOPTS} ${STATSOVERLAY} ${OUTLIERMASK} ${HUEOPTS} ${ROWOPTS} ${COLOPTS} ${STYLEOPTS} ${SIZEOPTS} ${PAGELIMITS} ${OTHERASIGOPTS}"
echo $OPTS
# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_mahal" --unitQuery="mahal" $OPTS
# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" $OPTS
# python -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral" --unitQuery="lfp" $OPTS
#
python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, pedalSizeCat" $HUEOPTS $OUTLIERMASK
python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, pedalSizeCat" $HUEOPTS $OUTLIERMASK
# python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp_CAR_scaled" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR $PAGELIMITS --groupPagesBy="electrode, pedalMovementCat, pedalDirection, pedalSizeCat" $HUEOPTS $OUTLIERMASK
# python -u './plotAlignedAsigsTopo.py' --inputBlockSuffix="lfp_spectral" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $TIMEWINDOWOPTS $BLOCKSELECTOR --groupPagesBy="electrode, pedalMovementCat, pedalDirection, pedalSizeCat, freqBandName" $HUEOPTS $OUTLIERMASK
#
# python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral_fa_mahal" --unitQuery="mahal" $OPTS
# python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral_mahal" --unitQuery="mahal" $OPTS
# python3 -u './plotAlignedAsigsV1.py' --inputBlockSuffix="lfp_CAR_spectral_fa" --unitQuery="factor" $OPTS
