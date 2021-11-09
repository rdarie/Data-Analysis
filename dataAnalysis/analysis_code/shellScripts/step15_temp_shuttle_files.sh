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
#  # exps=(exp201901251000 exp201901261000 exp201901271000 exp201902031100 exp201902041100 exp201902051100 exp202101201100 exp202101211100 exp202101221100 exp202101251100 exp202101271100 exp202101281100 exp202102021100)
#  # exps=(exp201901251000 exp201901261000 exp201901271000 exp202101271100 exp202101281100)
exps=(exp202101271100)
for EXP in "${exps[@]}"
do
  echo "step 15 cleanup, on $EXP"
  # DELETE these items
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*.html' --fromScratchToData --deleteItems --filesIncluded ${KEYPROMPT}
  #
  # scratch to data
  #
  # #after step 1

  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*.nix' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='utah*_chunkingInfo.json' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='*_synchFun.json' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Session*.nix' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_analyze.nix' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}
  python './shuttleFilesToFromScratch.py' --exp=$EXP --searchTerm='Block*_ins.nix' --fromScratchToData --moveItems --filesIncluded ${KEYPROMPT}

  # examine folder sizes
  #
  python './shuttleFilesToFromScratchV2.py' --exp=$EXP --printFolderSize
done