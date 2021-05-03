#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=8
#SBATCH --mem=127G

# Specify a job name:
#SBATCH -J pls_motion_lfp

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-pls_motion_lfp.out
#SBATCH -e ../../batch_logs/%j-%a-pls_motion_lfp.out

# Specify account details
#SBATCH --account=carney-dborton-condo

# Request custom resources
#SBATCH --array=2
SLURM_ARRAY_TASK_ID=2
source ./shellScripts/run_pca_calc_aligned_motion_preamble.sh
BLOCKSELECTOR="--blockIdx=${SLURM_ARRAY_TASK_ID} --processAll"

# iterator "b", -50 to 300 msec after start
python -u './calcOrdinaryLeastSquares.py' --loadFromFrames --iteratorSuffix='b' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="limbState" --estimatorName='ols' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './processOrdinaryLeastSquares.py' --fullEstimatorName='ols_lfp_CAR_spectral_to_limbState_b_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName='ols_lfp_CAR_spectral_to_limbState_b_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting

# iterator "c", 300 to 650 msec after start
python -u './calcOrdinaryLeastSquares.py' --loadFromFrames --iteratorSuffix='c' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="limbState" --estimatorName='ols' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './processOrdinaryLeastSquares.py' --fullEstimatorName='ols_lfp_CAR_spectral_to_limbState_c_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName='ols_lfp_CAR_spectral_to_limbState_c_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting

# iterator "a", movement start to stop
# python -u './calcGridSearchSingleTarget.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="limbState" --estimatorName='enr' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './calcGridSearchSingleTarget.py' --loadFromFrames --iteratorSuffix='a' --lhsBlockSuffix="lfp" --unitQueryLhs="lfp_CAR_spectral" --rhsBlockSuffix="rig" --unitQueryRhs="jointAngularVelocity" --estimatorName='enr' --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting --verbose=2
# python -u './processOrdinaryLeastSquares.py' --fullEstimatorName='enr_lfp_CAR_spectral_to_limbState_a_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
# python -u './plotOrdinaryLeastSquares.py' --fullEstimatorName='enr_lfp_CAR_spectral_to_limbState_a_L_starting' --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --plotting
