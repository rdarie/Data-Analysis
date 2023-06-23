#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J s14c_temp_mahal_dist_rauc_2019

# Specify an output file
#SBATCH -o ../../batch_logs/covariance/s14c_temp_mahal_dist_rauc_2019.out
#SBATCH -e ../../batch_logs/covariance/s14c_temp_mahal_dist_rauc_2019.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

# SLURM_ARRAY_TASK_ID=2
# exps=(201901_25 201902)
# exps=(202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)

exps=(202102_02)
for A in "${exps[@]}"; do
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_motion_preamble.sh
  #
  ITERATOR="ma"
  ESTIMATOR='mahal_ledoit'
  BLOCKSELECTOR="--blockIdx=2 --processAll"
  #
  # SELECTIONLIST="laplace_spectral_baseline, laplace_baseline"
  SELECTIONLIST="laplace_spectral_scaled, laplace_scaled, laplace_spectral_scaled_mahal_ledoit, laplace_scaled_mahal_ledoit"
  EXPLIST_RUPERT="exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100"
  EXPLIST_MURDOC="exp201901271000, exp201902031100"
  # exp201901271000, exp201902031100
  # exp202101201100, exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100
  # exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100
  # exp202101281100, exp202102021100, exp202101211100,

  EXPLIST_RUPERT="exp202101271100"
  EXPLIST_MURDOC="exp201901271000"
  SELECTIONLIST="lfp_spectral, lfp"
  #######
  ## plotSuffixes=(all move_E09 E04 rest_stim outbound_stim outbound_no_move best_three)
  #
  python -u "./plotSignalRecruitmentAcrossExpV5.py" --plotSuffix=all --freqBandGroup=1 --expList="${EXPLIST_RUPERT}" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
  python -u "./plotSignalRecruitmentAcrossExpV5.py" --plotSuffix=all --freqBandGroup=1 --expList="${EXPLIST_MURDOC}" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
  #
  freqBandGroupings=(1)
  for fbg in "${freqBandGroupings[@]}"; do
    echo $fbg
    # python -u "./plotSignalRecruitmentAcrossExpV5.py" --plotSuffix=move_E09       --freqBandGroup=$fbg --expList="exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
    # python -u "./plotSignalRecruitmentAcrossExpV5.py" --plotSuffix=return_E04     --freqBandGroup=$fbg --expList="exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
    # python -u "./plotSignalRecruitmentAcrossExpV5.py" --plotSuffix=outbound_E04   --freqBandGroup=$fbg --expList="exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100" --selectionList="${SELECTIONLIST}" $BLOCKSELECTOR $WINDOW $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR $ALIGNQUERY
  done
  ### all E04
  plotSuffixes=(E04)
  for fbg in "${freqBandGroupings[@]}"; do
    for PS in "${plotSuffixes[@]}"; do
      echo $PS
      # python -u "./compareSignalCovarianceMatricesAcrossExp.py" --freqBandGroup=$fbg --plotSuffix=$PS --expList="exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100" --targetList="laplace_spectral_scaled, laplace_scaled" --iteratorSuffixList="ca, cb, ccs, ccm" --estimatorName="mahal_ledoit" --datasetPrefix="Block_XL_df" $BLOCKSELECTOR $ANALYSISFOLDER $ALIGNFOLDER $UNITSELECTOR
    done
  done
done
