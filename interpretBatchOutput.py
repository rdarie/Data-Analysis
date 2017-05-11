import argparse
import re
import pandas as pd

try:
    parser = argparse.ArgumentParser()
    #pdb.set_trace()
    parser.add_argument('--file', nargs='*')
    args = parser.parse_args()
    argFile = args.file
except:
    argFile = ['trainLDA_DownSampledSpectrum.out']

WinLenPattern = 'Preprocessing spectral data with a window length of (\d+\.\d+)'
StepLenPattern = 'seconds and a step length of (\d+\.\d+) seconds\n'
FScorePattern = 'F1 Score for \w+ was:'

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

results = {'winLen': [], 'stepLen': [], 'F1Score': []}
for fileName in argFile:
    with open(localDir + '/' + fileName, 'r') as f:
        # Read the file contents and generate a list with each line
        lines = f.readlines()

    lookForWS = True
    # There could be many instances of WinLen, StepLen
    # that all refer to the same estimator, because each file produces
    # a printout. only look at the first ones, then stop looking until
    # you find an F1 Score.

    for idx,line in enumerate(lines):
        # Regex applied to each line
        if lookForWS:
            match = re.search(WinLenPattern, line)
            if match:
                winLen = float(match.groups()[0])

            match = re.search(StepLenPattern, line)
            if match:
                stepLen = float(match.groups()[0])
                lookForWS = False

        match = re.search(FScorePattern, line)
        if match:
            F1Score = float(lines[idx + 1])
            results['F1Score'].append(F1Score)
            results['winLen'].append(winLen)
            winLen = np.nan
            results['stepLen'].append(stepLen)
            stepLen = np.nan
            lookForWS = True

results = pd.DataFrame(results)
