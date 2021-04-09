#!/bin/bash

module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg
module load anaconda/2020.02
module load mpi
module load opengl
module load qt/5.10.1
module load zlib/1.2.11

. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

/gpfs/home/rdarie/pycharm/pycharm-community-2020.2.3/bin/pycharm.sh