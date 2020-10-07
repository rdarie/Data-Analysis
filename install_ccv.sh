#!/bin/bash
GitRepoRoot="git://github.com/rdarie/"
#GitRepoRoot="git://github.com/neuromotion/"
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
)

module load git/2.10.2
module load gcc/4.9.4
module load leveldb openblas hdf5 protobuf ffmpeg
module load opengl
module load anaconda/3-5.2.0
module load mpi
module load qt/5.10.1

. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate

conda env create -f environment-linux.yml
chmod +x $HOME/anaconda/nda2/bin/*
source activate nda2
cd ..
#
pip install pyqt5==5.10.1 --user
pip install vg==1.6.1 --user
pip install git+git://github.com/G-Node/nixpy@v1.5.0b3 --user
pip install git+git://github.com/hector-sab/ttictoc@v0.4.1 --user
pip install git+git://github.com/raphaelvallat/pingouin@v0.3.3 --user
pip install git+git://github.com/lmcinnes/umap.git@0.5dev --user
#

for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    git checkout tags/ndav0.2
    python setup.py develop --user
    cd ..
done
#
cd Data-Analysis
python setup.py develop --user
