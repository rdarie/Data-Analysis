#!/bin/bash
GitRepoRoot="git://github.com/rdarie/"
#GitRepoRoot="git://github.com/neuromotion/"
RepoList=("pyglmnet" \
"analysis-tools" \
"python-neo" \
"tridesclous" \
"ephyviewer" \
"rcsanalysis" \
"peakutils" \
"seaborn" \
"spykesML" \
"Data-Analysis"\
)

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