#!/bin/bash

LOCALPATH=/Users/radudarie/Documents/Github/Data-Analysis
REMOTEPATH=rdarie@ssh.ccv.brown.edu:/gpfs/home/rdarie/nda2/Data-Analysis

scp -r $LOCALPATH %REMOTEPATH