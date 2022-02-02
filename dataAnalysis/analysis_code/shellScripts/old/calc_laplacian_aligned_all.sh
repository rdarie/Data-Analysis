#!/bin/bash

# Request runtime:
#SBATCH --time=32:00:00

# Default resources are 1 core with 2.8GB of memory.

# Request memory:
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem-per-cpu=48G
#SBATCH --hint=memory_bound

# Specify a job name:
#SBATCH -J lapl_calc_201902-04

# Specify an output file
#SBATCH -o ../../batch_logs/lapl_calc_201902-04-%a.out
#SBATCH -e ../../batch_logs/lapl_calc_201902-04-%a.out

# Specify account details
#SBATCH --account=carney-dborton-condo
#SBATCH --export=CCV_HEADLESS=1

# Request custom resources
#SBATCH --array=5

# SLURM_ARRAY_TASK_ID=1
exps=(201902_04)
for A in "${exps[@]}"
do
  echo "step 01b calc laplacian and spectral components, on $A"
  source ./shellScripts/run_exp_preamble_$A.sh
  source ./shellScripts/run_align_motion_preamble.sh
  
  ALIGNQUERY="--alignQuery=starting"
  echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
  python -u ./calcLaplacianFromTriggeredV3.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  python -u ./calcWaveletFeatures.py --inputBlockSuffix="laplace" --unitQuery="csd" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  source ./shellScripts/run_align_stim_preamble.sh
  
  ALIGNQUERY="--alignQuery=stimOn"
  echo "$WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS"
  python -u ./calcLaplacianFromTriggeredV3.py --inputBlockSuffix="lfp" --unitQuery="lfp" --outputBlockSuffix="laplace" --plotting --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
  #
  python -u ./calcWaveletFeatures.py --inputBlockSuffix="laplace" --unitQuery="csd" $VERBOSITY --exp=$EXP $WINDOW $ALIGNQUERY $ANALYSISFOLDER $ALIGNFOLDER $BLOCKSELECTOR $LAZINESS
done