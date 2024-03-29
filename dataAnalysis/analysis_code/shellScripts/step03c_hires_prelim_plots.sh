#!/bin/bash

# Request runtime:
#SBATCH --time=12:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --nodes=1
#SBATCH --mem=250G

# Specify a job name:
#SBATCH -J s03c_hires_plots_202101_27

# Specify an output file
#SBATCH -o ../../batch_logs/s03c_hires_plots_202101_27.out
#SBATCH -e ../../batch_logs/s03c_hires_plots_202101_27.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=999

#    SLURM_ARRAY_TASK_ID=2
#

# exps=(201902 201902_04 201902_05)
# exps=(201901_25 201901_26 201901_27 201902 201902_04 201902_05 202101_20 202101_21 202101_22 202101_25 202101_27 202101_28 202102_02)
exps=(202101_27)
for A in "${exps[@]}"
do
  echo "step 03b plots, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/calc_aligned_stim_raw_preamble.sh
  #
  BLOCKSELECTOR="--blockIdx=1 --processAll"
  # BLOCKSELECTOR="--blockIdx=1"
  iters=(pb pc)
  for ITERATOR in "${iters[@]}"
  do
    #--------#
    python -u './calcTestTrainSplit.py' $BLOCKSELECTOR --iteratorSuffix=$ITERATOR --loadFromFrames --inputBlockSuffix="rig" --unitQuery="rig" --selectionName='rig' --verbose --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $LAZINESS $TIMEWINDOWOPTS
    #--------#
    COMMONOPTS=" --iteratorSuffix=${ITERATOR} --loadFromFrames --exp=${EXP} ${WINDOW} ${ALIGNQUERY} ${ANALYSISFOLDER} ${ALIGNFOLDER} ${BLOCKSELECTOR} --plotting --verbose=2"
    #--------#
    python -u './assembleDataFrames.py' --resetHDF --inputBlockSuffix="lfp" --selectionName="lfp" $COMMONOPTS
    python -u './assembleDataFrames.py' --inputBlockSuffix="laplace" --selectionName="laplace" $COMMONOPTS
    python -u './assembleDataFrames.py' --inputBlockSuffix='rig' --selectionName='rig' $COMMONOPTS
    #--------#
    TARGET="lfp"
    #--------#
    python -u './calcAverageFeature.py' --estimatorName="column_average" --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}" --selectionName=$TARGET --exp=$EXP $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR --verbose=2 --plotting
    #--------#
    OPTS="--enableOverrides --exp=${EXP} ${BLOCKSELECTOR} ${ANALYSISFOLDER} ${WINDOW} ${ALIGNFOLDER} ${ALIGNQUERY}"
    #--------#
    python -u './plotSignalDataFrameV2.py' --plotSuffix="lfp_hf_illustration" --verbose=1 --selectionName="lfp_column_average" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
    python -u './plotSignalDataFrameV2.py' --plotSuffix="lfp_hf_illustration" --saveFeatureInfoHTML --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
    python -u './plotSignalDataFrameV2.py' --plotSuffix="laplace_hf_illustration" --saveFeatureInfoHTML --verbose=1 --selectionName="laplace" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
    python -u './plotSignalDataFrameV2.py' --plotSuffix="rig_hf_illustration" --verbose=1 --selectionName="rig" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
    ## python -u './plotSignalDataFrameV2.py' --plotSuffix="lfp_hf_topo_illustration" --verbose=1 --selectionName="lfp" $OPTS --datasetName="Block_${WINDOWTERM}_df_${ITERATOR}"
    #--------#
  done
done