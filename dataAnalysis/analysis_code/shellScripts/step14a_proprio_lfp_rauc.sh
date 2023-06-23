#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=300G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J s14a_mahal_dist_rauc_202101

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s14a_mahal_dist_rauc_202101.out
#SBATCH -e ../../batch_logs/covariance/s14a_mahal_dist_rauc_202101.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_25 201902
# # 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

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
  python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
  # --preScale
  COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
  python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="lfp" --selectionName="lfp" $COMMONOPTS
  python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_spectral" --selectionName="lfp_spectral" $COMMONOPTS
  python -u './assembleDataFrames.py'            --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
  # python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_${ESTIMATOR}" --selectionName="lfp_${ESTIMATOR}" $COMMONOPTS
  # python -u './assembleDataFrames.py'            --inputBlockSuffix="lfp_spectral_${ESTIMATOR}" --selectionName="lfp_spectral_${ESTIMATOR}" $COMMONOPTS
  #
  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_per_trial_illustration" --verbose=1 --selectionName="lfp_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_illustration" --verbose=1 --selectionName="lfp_${ESTIMATOR}" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  UNITSELECTOR="--unitQuery=mahal"
  ALIGNQUERYTERM="starting"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  TARGET="lfp_${ESTIMATOR}"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  TARGET="lfp_spectral_${ESTIMATOR}"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  TARGET="lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV2.py" --plotThePieces --plotTheAverage --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  TARGET="lfp_spectral"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  #
  python -u "./calcSignalRecruitmentV4.py" --iteratorSuffix=$ITERATOR --plotting --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER --loadFromFrames --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET
  python -u "./plotSignalRecruitmentV2.py" --plotThePieces --plotTheAverage --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_topo_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_auc_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_aucNoStim_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_auc_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_std_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ##
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_aucNoStim_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_auc_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_topo_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_std_topo_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_auc_illustration" --verbose=1 --selectionName="lfp_spectral_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_aucNoStim_illustration" --verbose=1 --selectionName="lfp_spectral_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_auc_illustration" --verbose=1 --selectionName="lfp_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_aucNoStim_illustration" --verbose=1 --selectionName="lfp_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="rig_per_trial_illustration" --verbose=1 --selectionName="rig" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
done