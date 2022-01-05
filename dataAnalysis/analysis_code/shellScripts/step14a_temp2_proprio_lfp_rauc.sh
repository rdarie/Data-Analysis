#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=125G

# Specify a job name:
#SBATCH -J s14a_temp_mahal_dist_rauc_201901_25

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s14a_temp_mahal_dist_rauc_201901_25.out
#SBATCH -e ../../batch_logs/covariance/s14a_temp_mahal_dist_rauc_201901_25.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_25 201902_03 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 12 lfp rauc, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  #
  ITERATOR="ma"
  ESTIMATOR='mahal_ledoit'
  #
  # #python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  # --preScale
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  # #python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="laplace_scaled_${ESTIMATOR}" --selectionName="laplace_scaled_${ESTIMATOR}" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral_scaled_${ESTIMATOR}" --selectionName="laplace_spectral_scaled_${ESTIMATOR}" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_scaled" --selectionName="laplace_scaled" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral_scaled" --selectionName="laplace_spectral_scaled" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_baseline" --selectionName="laplace_baseline" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix="laplace_spectral_baseline" --selectionName="laplace_spectral_baseline" $COMMONOPTS
  # #python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  #
  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  # ## #python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_per_trial_illustration" --verbose=1 --selectionName="laplace_scaled_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  # ## #python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="laplace_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  # ## #python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="laplace_scaled_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  UNITSELECTOR="--unitQuery=mahal"
  ALIGNQUERYTERM="starting"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  TARGET="laplace_spectral_scaled_${ESTIMATOR}"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  # python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV5.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  TARGET="laplace_scaled_${ESTIMATOR}"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  # python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV5.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  # #
  TARGET="laplace_spectral_baseline"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  # --plotTheAverage
  # python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV5.py" --plotThePieces --plotTopoEffectSize --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  # #
  TARGET="laplace_baseline"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  # python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV5.py" --plotThePieces --plotTopoEffectSize --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  # #
  ######
  python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="laplace_scaled_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="laplace_spectral_scaled_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="laplace_baseline" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="laplace_spectral_baseline" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  # python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration_topo" --verbose=1 --selectionName="laplace_scaled" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
done