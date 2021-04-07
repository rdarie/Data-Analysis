#!/bin/bash
GitRepoRoot="git://github.com/rdarie/"

RepoList=(\
"seaborn" \
"python-neo" \
"tridesclous" \
"ephyviewer" \
"elephant" \
"pyglmnet" \
"analysis-tools" \
"rcsanalysis" \
"peakutils" \
"umap" \
"kCSD-python"
)

module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg fftw scons
module load anaconda/2020.02
module load mpi
# module load opengl
module load qt/5.10.1
module load zlib/1.2.11

. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

echo "Please wait for conda to install the environment..."
conda env create -f environment-open.yml

echo "Please check if installation was successful"
read FILLER
chmod +x $HOME/anaconda/nda2/bin/*
conda activate nda2
cd ..
export PYTHONPATH="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages"
#
pip install pyqt5==5.10.1 --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages"
pip install vg==1.6.1 --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages"
#
pip install git+git://github.com/G-Node/nixpy@v1.5.0b3 --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
pip install git+git://github.com/hector-sab/ttictoc@v0.4.1 --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
pip install git+git://github.com/raphaelvallat/pingouin@v0.3.3 --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
pip install git+git://github.com/melizalab/libtfr --target="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
#
for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    # git checkout tags/ndav0.3
    python setup.py develop --install-dir="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
    cd ..
done
#
cd Data-Analysis
python setup.py develop --install-dir="/users/rdarie/anaconda/nda2/lib/python3.7/site-packages" --no-deps
