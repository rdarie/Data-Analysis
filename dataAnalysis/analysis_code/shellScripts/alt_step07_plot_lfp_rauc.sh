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
#SBATCH -J s05_plot_lfp_rauc

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s05_plot_lfp_rauc_201901_27.out
#SBATCH -e ../../batch_logs/covariance/s05_plot_lfp_rauc_201901_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_25 201902
# # 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(201901_27)
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

  OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER}"
  #
  UNITSELECTOR="--unitQuery=mahal"
  ALIGNQUERYTERM="starting"
  ALIGNQUERY="--alignQuery=${ALIGNQUERYTERM}"
  #
  ## ## TARGET="lfp_${ESTIMATOR}"
  ## ## INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  ## ## python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  ## ## #
  ## ## TARGET="lfp_spectral_${ESTIMATOR}"
  ## ## INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  ## ## python -u "./plotSignalRecruitmentV2.py" --plotThePieces --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  ## ## #
  TARGET="lfp"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  # python -u "./plotSignalRecruitmentV2.py" --plotThePieces --plotTheAverage --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  TARGET="lfp_spectral"
  INPUTBLOCKNAME="--inputBlockSuffix=${TARGET} --inputBlockPrefix=Block"
  # python -u "./plotSignalRecruitmentV2.py" --plotThePieces --plotTheAverage --exp=$EXP $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $INPUTBLOCKNAME $UNITSELECTOR $ALIGNQUERY
  #
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_topo_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_topo_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"

  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_auc_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_aucNoStim_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_auc_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_std_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ##
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_aucNoStim_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_auc_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_spectral_std_topo_illustration" --verbose=1 --selectionName="lfp_spectral" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  #
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_auc_illustration" --verbose=1 --selectionName="lfp_spectral_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_aucNoStim_illustration" --verbose=1 --selectionName="lfp_spectral_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_auc_illustration" --verbose=1 --selectionName="lfp_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="mahal_aucNoStim_illustration" --verbose=1 --selectionName="lfp_mahal_ledoit" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
  ## ## python -u './plotSignalDataFrameV2.py' --plotSuffix="rig_per_trial_illustration" --verbose=1 --selectionName="rig" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
done