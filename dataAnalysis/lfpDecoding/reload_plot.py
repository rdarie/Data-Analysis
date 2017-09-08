from dataAnalysis.helperFunctions.helper_functions import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'None')

args = parser.parse_args()
argFile = args.file
reloadPlot(filePath = argFile)
