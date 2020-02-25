#!/bin/bash
#  10: Calculate align Times
# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=54
#SBATCH --tasks=54
#SBATCH --tasks-per-node=1
#SBATCH --mem=48G

# Specify a job name:
#SBATCH -J glmFull_20190127

# Specify an output file
#SBATCH -o ../batch_logs/%j-glmFull_20190127.stdout
#SBATCH -e ../batch_logs/%j-glmFull_20190127.errout

# Specify account details
#disableSBATCH -p bigmem
#SBATCH --account=bibs-dborton-condo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=radu_darie@brown.edu

# EXP="exp201901211000"
EXP="exp201901271000"
# ESTIMATOR="glm_20msec"
ESTIMATOR="glm_30msec"

# UNITSELECTOR=""
UNITSELECTOR="--selector=_minfrmaxcorr"
# UNITSELECTOR="--selector=_minfrmaxcorrminamp"
# UNITSELECTOR="--selector=_glm_30msec_long_midPeak_maxmod"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda
python --version

module load mpi
srun --mpi=pmi2 python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --maskOutlierTrials --estimatorName=$ESTIMATOR --verbose --plotting --attemptMPI
# python3 -u ./calcUnitGLMToAsig.py --exp=$EXP --processAll --inputBlockName="raster" --unitQuery="raster" $UNITSELECTOR --secondaryBlockName="rig" --alignQuery="midPeak" --maskOutlierTrials --estimatorName=$ESTIMATOR --verbose --plotting --debugging --dryRun

# debugging
# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --processAll --alignQuery="midPeak" --estimatorName=$ESTIMATOR --lazy --verbose --debugging --plottingIndividual --plottingOverall --makePredictionPDF

# full
# python3 ./evaluateUnitGLMToAsig.py --exp=$EXP --processAll --alignQuery="midPeak" --estimatorName=$ESTIMATOR --lazy --verbose --plottingIndividual --plottingOverall --makePredictionPDF