#!/bin/bash
# 
# Request runtime:
#SBATCH --time=1:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=16
#SBATCH --tasks=16
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J glmCollation_20190127

# Specify an output file
#SBATCH -o ../../batch_logs/%j-glmCollation_20190127.out
#SBATCH -e ../../batch_logs/%j-glmCollation_20190127.out

# Specify account details
#disableSBATCH -p bigmem
#SBATCH --account=carney-dborton-condo
#disableSBATCH --mail-type=ALL
#disableSBATCH --mail-user=radu_darie@brown.edu

# EXP="exp201901211000"
EXP="exp201901271000"
ESTIMATOR="glm_20msec"
# ESTIMATOR="glm_50msec"

# UNITSELECTOR=""
UNITSELECTOR="--selector=_minfrmaxcorr"
# UNITSELECTOR="--selector=_minfrmaxcorrminamp"
# UNITSELECTOR="--selector=_glm_30msec_long_midPeak_maxmod"

module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

# parallel collation
# python3 -u ./collateUnitGLMToAsigParallel.py --exp=$EXP --processAll --alignQuery="midPeak" --estimatorName=$ESTIMATOR --lazy --verbose --debugging
module load mpi
srun --mpi=pmi2 python3 -u ./collateUnitGLMToAsigParallel.py --exp=$EXP --processAll --alignQuery="midPeak" --estimatorName=$ESTIMATOR --lazy --verbose --debugging --attemptMPI
