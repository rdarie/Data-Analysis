#!/bin/bash

# 06b: Preprocess the INS Recording
# Request 24 hours of runtime:
#SBATCH --time=6:00:00

# Default resources are 1 core with 2.8GB of memory.
# Request custom resources
#SBATCH --nodes=1
#SBATCH --mem=24G

# Specify a job name:
#SBATCH -J ins_preproc

# Specify an output file
#SBATCH -o ../../batch_logs/%j-%a-ins_preproc.out
#SBATCH -e ../../batch_logs/%j-%a-ins_preproc.out

# Specify account details
#SBATCH --account=bibs-dborton-condo

# Request custom resources
#SBATCH --array=1,2,3

####

# EXP="exp201901251000"
# has 1 minirc 2 motionsba
# EXP="exp201901261000"
# has 1-3 motion 4 minirc
# EXP="exp201901271000"
# has 1-4 motion 5 minirc
#
# EXP="exp201901311000"
# has 1-4 motion 5 minirc
# EXP="exp201902010900"
#  has 1-4 motion 5 minirc
# EXP="exp201902021100"
# has 3-5 motion 6 minirc; blocks 1 and 2 were bad;
# EXP="exp201902031100"
# has 1-4 motion 5 minirc;
# EXP="exp201902041100"
# has 1-4 motion 5 minirc;
# EXP="exp201902051100"
# has 1-4 motion
########
# EXP="exp202101201100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202101211100"
# has 1 minirc 2,3 motion+stim 4 motionOnly
# EXP="exp202101221100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202101251100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202101271100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202101281100"
# has 1 minirc 2 motion+stim 3 motionOnly
# EXP="exp202102021100"
# has 1 minirc 2 motion+stim 3 motionOnly

# ANALYSISSELECTOR="--analysisName=emgHiRes"
# ANALYSISSELECTOR="--analysisName=emgLoRes"
# ANALYSISSELECTOR="--analysisName=lfpFullRes"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

source activate nda2
python --version

# radujobinfo
# Info about jobs for user 'rdarie' submitted since 2021-08-31T00:00:00
# Use option '-S' for a different date
#  or option '-j' for a specific Job ID.

WINDOW="XL"
# KEYPROMPT="--requireKeypress"
KEYPROMPT=""
# exps=(exp201901251000 exp201901261000 exp201901271000 exp201902031100 exp201902041100 exp201902051100) 
#  # exps=(exp202101201100 exp202101211100 exp202101221100 exp202101251100 exp202101271100 exp202101281100 exp202102021100)
exps=(exp202101211100)
for EXP in "${exps[@]}"
do
  echo "step 16, restore from processed, on $EXP"
  #
  # scratch to data
  #
  # after step 1
  # #
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.nix' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_chunkingInfo.json' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_synchFun.json' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Session*.nix' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins.nix' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # ## #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze.nix' --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # after step 2
  # 
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_lfp_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_lfp_CAR_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  #
  # after step 3
  # 
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block_*_pa_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block_*_pa.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block_*_pa_*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # 
  # # after step 4
  # 
  # #BLOCK_ID_NO="0*"
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_spectral_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # 
  # #BLOCK_ID_NO="_*"
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_na_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_na.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_na_*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # 
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="scaled_*_na_*.joblib" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="scaled_*_na_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # 
  # after step 5
  # after step 6
  # #BLOCK_ID_NO="_*"
  # #iters=(ra)
  # #for ITER in "${iters[@]}"
  # #do
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_cvIterators.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  #
  # #  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*laplace_scaled*.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*laplace_scaled*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*laplace_spectral_scaled*.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*laplace_spectral_scaled*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*rig*.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*rig*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="ols_*_${ITER}.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="ols_*_${ITER}_*.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="ols_*_${ITER}_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="ols_*_${ITER}_*" --fromDataToScratch --foldersIncluded ${KEYPROMPT}
  # #done
  # #
  # after step 10
  # #
  # #BLOCK_ID_NO="_*"
  # #iters=(ca cb ccm ccs)
  # #for ITER in "${iters[@]}"
  # #do
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}_*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #done
  # #
  # #after step 11
  # #
  BLOCK_ID_NO="0*"
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_scaled_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_spectral_scaled_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_baseline_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_laplace_spectral_baseline_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # #after step 12
  # #BLOCK_ID_NO="_*"
  # #ITER="cd"
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="baseline_*_cd_*.joblib" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="baseline_*_cd_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_${ITER}_*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # #ITER="ma"
  # #BLOCK_ID_NO="_*"
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}*${ITER}_*_meta.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # #iters=(ca cb ccm ccs)
  # #for ITER in "${iters[@]}"
  # #do
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="mahal_ledoit_*_${ITER}_*.joblib" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="mahal_ledoit_*_${ITER}_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="mahal_ledoit_*_${ITER}_*.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #done
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block${BLOCK_ID_NO}_mahal_ledoit_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_rauc_iterator_*.pickle" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="*_covarianceMatrixCalc.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="*_rauc.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # misc
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_rig_${WINDOW}.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_${WINDOW}_outliers.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_${WINDOW}_outliers.h5" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_${WINDOW}_outliers.csv" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_epochs.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # #
  # #python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm="Block*_${WINDOW}_*viewable.nix" --fromDataToScratch --filesIncluded ${KEYPROMPT}
  # 
  # 
  # examine folder sizes
  python './shuttleFilesToFromScratchV2.py' --exp=$EXP --printFolderSize
done