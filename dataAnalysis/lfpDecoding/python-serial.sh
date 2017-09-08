#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=0:10:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB):
#SBATCH --mem=64G
#SBATCH -n 12

# Specify a job name:
#SBATCH -J PythonJob

# Run a matlab function called 'foo.m' in the same directory as this batch script.
python3 mainSpikes.py
