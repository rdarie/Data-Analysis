#!/bin/bash

# EXP="exp201901261000"
# EXP="exp202010271200"
# EXP="exp202011161100"
# EXP="exp202011201100"
# EXP="exp202011271100"
# EXP="exp202012071100"
# EXP="exp202012081200"
# EXP="exp202012091200"
# EXP="exp202012101100"
# EXP="exp202012111100"
# EXP="exp202012121100"
# EXP="exp202012151200"
# EXP="exp202012161200"
# EXP="exp202012171200"
# EXP="exp202012181200"
EXP="exp202101061100"
EXP="exp202101111100"

module load anaconda/2020.02
. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

