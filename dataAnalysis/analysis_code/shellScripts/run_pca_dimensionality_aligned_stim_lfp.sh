#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pca_calc_stim_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pca_calc_stim_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-pca_calc_stim_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2

SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_stim_preamble.sh

# python -u './calcLaplacianFromTriggered.py' --plotting --recalcKCSDCV --useKCSD --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="kcsd_triggered" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK $LAZINESS

# python -u './calcPCAinChunks.py' --inputBlockSuffix="kcsd" --unitQuery="lfp" --estimatorName="pca_lapl" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK
python -u './calcSignalDimensionality.py' --inputBlockSuffix="lfp" --unitQuery="lfp" --estimatorName="pca_lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './calcSignalDimensionality.py' --inputBlockSuffix="lfp_CAR" --unitQuery="lfp" --estimatorName="pca_car" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
# python -u './calcSparsePCA.py' --inputBlockSuffix="kcsd" --unitQuery="lfp" --estimatorName="sparse_pca_lapl" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $OUTLIERMASK
# python -u './applyEstimatorToTriggered.py' --inputBlockSuffix="kcsd" --estimatorName="pca_lapl" --unitQuery="lfp" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
# python -u './calcSpectralFeatures.py' --inputBlockSuffix="kcsd_pca" --unitQuery="pca" --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR
