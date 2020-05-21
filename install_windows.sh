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

# conda env create -f environment-windows.yml
# chmod +x $HOME/anaconda/nda2/bin/*
# conda activate nda2
# cd ..
#
pip install pyqt5==5.10.1 --user
pip install vg==1.6.1 --user
pip install mpi4py==3.0.3 --user
pip install git+git://github.com/G-Node/nixpy@v1.5.0b3 --user
pip install git+git://github.com/hector-sab/ttictoc@v0.4.1 --user
pip install git+git://github.com/raphaelvallat/pingouin@v0.3.3 --user
#

for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    # git checkout tags/ndav0.2
    python setup.py develop --user
    cd ..
done
#
cd Data-Analysis
python setup.py develop --user
