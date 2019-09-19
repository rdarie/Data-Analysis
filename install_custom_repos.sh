#!/bin/bash
GitRepoRoot="git://github.com/rdarie/"
#GitRepoRoot="git://github.com/neuromotion/"
RepoList=(\
"python-neo" \
"tridesclous" \
"ephyviewer" \
"pyglmnet" \
"analysis-tools" \
"rcsanalysis" \
"peakutils" \
"seaborn" \
"spykesML" \
"Data-Analysis"\
)
#
git clone git://github.com/NeuralEnsemble/elephant.git
cd elephant
python3 setup.py develop --user
cd ..
#
git clone git://github.com/G-Node/nixpy.git
cd nixpy
git checkout tags/v1.5.0b3
python3 setup.py develop --user
cd ..
#
pip install pyqt5==5.10.1 --user
#
cd ../
echo $(pwd)
for i in ${RepoList[*]}; do
    echo $GitRepoRoot$i".git"
    git clone $GitRepoRoot$i".git"
    cd $i
    python3 setup.py develop --user
    cd ../
done
