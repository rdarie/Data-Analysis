#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=24:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=12
#SBATCH --tasks=12
#SBATCH --tasks-per-node=1
#SBATCH --mem=72G

# Specify a job name:
#SBATCH -J glmFull_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-glmFull_20190127.out
#SBATCH -e ../../batch_logs/%j-glmFull_20190127.out

# Specify account details
#disableSBATCH -p bigmem
#SBATCH --account=carney-dborton-condo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=radu_darie@brown.edu

# EXP="exp201901211000"
EXP="exp201901271000"
# ESTIMATOR="glm_20msec"
ESTIMATOR="glm_20msec"

# UNITSELECTOR=""
UNITSELECTOR="--selector=_minfrmaxcorr"
# UNITSELECTOR="--selector=_minfrmaxcorrminamp"
# UNITSELECTOR="--selector=_glm_30msec_long_midPeak_maxmod"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

module load mpi
srun --mpi=pmi2 python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --maskOutlierBlocks --estimatorName=$ESTIMATOR --verbose --plotting --attemptMPI
# python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --maskOutlierBlocks --estimatorName=$ESTIMATOR --verbose --plotting --debugging --dryRun
# python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --maskOutlierBlocks --estimatorName=$ESTIMATOR --plotting --verbose