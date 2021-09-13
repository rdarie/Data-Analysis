#!/bin/bash

module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg fftw scons
module load anaconda/2020.02
module load mpi
module load opengl
# module load opengl/mesa-12.0.6
module load qt/5.10.1
module load zlib/1.2.11
module unload python

. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate
source activate nda2
python --version

export OUTDATED_IGNORE=1

/gpfs/home/rdarie/pycharm/pycharm-community-2020.2.3/bin/pycharm.sh
