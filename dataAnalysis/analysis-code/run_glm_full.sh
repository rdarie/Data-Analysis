#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=250:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=36
#SBATCH --tasks=36
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J glmFull_20190127

# Specify an output file
#SBATCH -o ../batch_logs/%j-glmFull_20190127.stdout
#SBATCH -e ../batch_logs/%j-glmFull_20190127.errout

# Specify account details
#SBATCH --account=bibs-dborton-condo

# EXP="exp201901211000"
EXP="exp201901271000"
# ESTIMATOR="glm_20msec_hires"
ESTIMATOR="glm_1msec"

# UNITSELECTOR=""
# UNITSELECTOR="--selector=_minfrmaxcorr"
UNITSELECTOR="--selector=_minfrmaxcorrminamp"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

module load mpi
srun --mpi=pmi2 python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --estimatorName=$ESTIMATOR --verbose --attemptMPI
# python3 ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --estimatorName=$ESTIMATOR --verbose --plotting

# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --processAll --alignQuery="midPeak" --estimatorName=$ESTIMATOR --lazy --verbose

# python3 ./applyEstimatorToTriggered.py --exp=$EXP --processAll --window="long" --alignQuery="midPeak" --estimator=$ESTIMATOR --lazy --profile --verbose
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim100HzCW" --rowName="pedalSizeCat"
# python3 ./plotAlignedAsigs.py --exp=$EXP --processAll --window="long" --inputBlockName=$BLOCKNAME --unitQuery="all" --alignQuery="midPeakWithStim50HzCW" --rowName="pedalSizeCat"