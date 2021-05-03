#!/bin/bash

LOCALROOT=/Users/radudarie/Documents/Github/Data-Analysis/
REMOTEROOT=rdarie@ssh.ccv.brown.edu:/gpfs/home/rdarie/nda2/Data-Analysis/

# scp -r $LOCALPATH $REMOTEPATH
cd $LOCALROOT
x=( $(git ls-files -m) );
for item in "${x[@]}"; {
    LOCALPATH=($LOCALROOT$item);
    REMOTEPATH=($REMOTEROOT$item);
    scp -r $LOCALPATH $REMOTEPATH
    }