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
"umap"
)

module load git/2.10.2
module load gcc/8.3
module load leveldb lapack openblas llvm hdf5 protobuf ffmpeg
module load opengl
module load anaconda/2020.02
module load mpi
module load qt/5.10.1

. /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate

conda env create -f environment-open.yml
chmod +x $HOME/anaconda/nda2/bin/*
source activate nda2
cd ..
#
pip install git+git://github.com/G-Node/nixpy@v1.5.0b3 --user --no-deps
pip install git+git://github.com/hector-sab/ttictoc@v0.4.1 --user --no-deps
pip install git+git://github.com/raphaelvallat/pingouin@v0.3.3 --user --no-deps
#

for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    git checkout tags/ndav0.3
    python setup.py develop --user --no-deps
    cd ..
done
#
cd Data-Analysis
python setup.py develop --user --no-deps
