"""
Usage:
    temp.py [options]

Options:
    --exp=exp                                   which experimental day to analyze
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --processAll                                process entire experimental day? [default: False]
    --lazy                                      load from raw, or regular? [default: False]
    --verbose                                   print diagnostics? [default: False]
    --exportToDeepSpine                         look for a deepspine exported h5 and save these there [default: False]
    --plotting                                  plot out the correlation matrix? [default: True]
    --showPlots                                 show the plots? [default: False]
    --analysisName=analysisName                 append a name to the resulting blocks? [default: default]
    --inputBlockSuffix=inputBlockSuffix         filename for inputs [default: fr]
    --maskOutlierBlocks                         delete outlier trials? [default: False]
    --window=window                             process with short window? [default: long]
    --unitQuery=unitQuery                       how to restrict channels if not supplying a list? [default: fr]
    --alignQuery=alignQuery                     query what the units will be aligned to? [default: midPeak]
    --alignFolderName=alignFolderName           append a name to the resulting blocks? [default: motion]
    --selector=selector                         filename if using a unit selector
    --amplitudeFieldName=amplitudeFieldName     what is the amplitude named? [default: nominalCurrent]
"""
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
if arguments['plotting']:
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.use('QT5Agg')   # generate postscript output
    # matplotlib.use('Agg')   # generate postscript output
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set()
    sns.set_color_codes("dark")
    sns.set_context("notebook")
    sns.set_style("dark")
# from tqdm import tqdm
import pdb
import os
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, StandardScaler

from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from sklearn.preprocessing import scale, robust_scale
from dask.distributed import Client, LocalCluster

#
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName'])
if not os.path.exists(alignSubFolder):
    os.makedirs(alignSubFolder, exist_ok=True)
calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
if not os.path.exists(calcSubFolder):
    os.makedirs(calcSubFolder, exist_ok=True)

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockSuffix'], arguments['window']))
resultPath = os.path.join(
    calcSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockSuffix'], arguments['window']))
# e.g. resultPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202006171300-Peep/emgLoRes/stim/_emg_XS.nix'
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    removeFuzzyName=False,
    decimate=1, windowSize=(0, 300e-3),
    metaDataToCategories=False,
    getMetaData=[
        'RateInHz', 'feature', 'electrode',
        arguments['amplitudeFieldName'], 'stimPeriod',
        'pedalMovementCat', 'pedalSizeCat', 'pedalDirection',
        'stimCat', 'originalIndex', 'segment', 't'],
    transposeToColumns='bin', concatOn='index',
    verbose=False, procFun=None))
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['outlierTrials'] = ash.processOutlierTrials(
    scratchFolder, prefix, **arguments)

if __name__ == "__main__":
    testVar = None
    conditionNames = [
        'electrode',
        'RateInHz', 'nominalCurrent']
    groupBy = ['feature'] + conditionNames
    # daskComputeOpts = {}
    daskComputeOpts = dict(
        # scheduler='threads'
        scheduler='processes'
        # scheduler='single-threaded'
        )
    # resultMeta = {
    #     'noiseCeil': float,
    #     'noiseCeilStd': float,
    #     'covariance': float,
    #     'covarianceStd': float,
    #     'mse': float,
    #     'mseStd': float
    #     }
    useCachedResult = False
    if not (useCachedResult and os.path.exists(resultPath)):
        print('loading {}'.format(triggeredPath))
        dataReader, dataBlock = preproc.blockFromPath(
            triggeredPath, lazy=arguments['lazy'])
        dataDF = preproc.alignedAsigsToDF(
            dataBlock, **alignedAsigsKWargs)
        egFeatName = dataDF.index.get_level_values('feature').unique()[0]
        breakDownData, breakDownText, fig, ax = asp.printBreakdown(
            dataDF.xs(egFeatName, level='feature', drop_level=False),
            'RateInHz', 'electrode', 'nominalCurrent')
        funKWArgs = dict(
                tBounds=None,
                plotting=False, iterMethod='half',
                corrMethod='pearson', maxIter=100)
        exampleOutput = pd.DataFrame(
            {
                'noiseCeil': float(1),
                'noiseCeilStd': float(1),
                'covariance': float(1),
                'covarianceStd': float(1),
                'mse': float(1),
                'mseStd': float(1)}, index=[0])
        resDF = ash.splitApplyCombine(
            dataDF, fun=hf.noiseCeil, resultPath=resultPath,
            funArgs=[], funKWArgs=funKWArgs,
            rowKeys=groupBy, colKeys=testVar, useDask=True,
            daskPersist=True, daskProgBar=True, daskResultMeta=None,
            daskComputeOpts=daskComputeOpts, nPartitionMultiplier=2,
            retainInputIndex=True)
        # resDF = ash.applyFunGrouped(
        #     dataDF,
        #     groupBy, testVar,
        #     fun=noiseCeil, funArgs=[],
        #     funKWargs=dict(
        #         plotting=arguments['plotting'],
        #         iterMethod='half', maxIter=1e2),
        #     resultNames=resultNames, plotting=False)
        for rN in resDF.columns:
            resDF[rN].to_hdf(resultPath, rN, format='fixed')
        noiseCeil = resDF['noiseCeil']
        covar = resDF['covariance']
        mse = resDF['mse']
    else:
        noiseCeil = pd.read_hdf(resultPath, 'noiseCeil')
        covar = pd.read_hdf(resultPath, 'covariance')
        mse = pd.read_hdf(resultPath, 'mse')
        resDF = pd.concat({
            'noiseCeil': noiseCeil,
            'covariance': covar,
            'mse': mse,
            }, axis='columns')
    for cN in ['covariance', 'mse']:
        robScaler = RobustScaler(quantile_range=(5, 95))
        # inputDF = resDF[cN].unstack(level='feature')
        #
        robScaler.fit(resDF.loc[resDF[cN].notna(), cN].to_numpy().reshape(-1, 1))
        preScaled = (robScaler.transform(resDF[cN].to_numpy().reshape(-1, 1)))
        resDF[cN + '_q_scale'] = pd.Series(
            preScaled.squeeze(),
            index=resDF[cN].index)
        scaledMask = np.abs(preScaled.squeeze()) < 2
        # scaledMask = pd.Series(
        #     np.abs(preScaled.squeeze()) < 2,
        #     index=resDF[cN].index)
        mmScaler = MinMaxScaler()
        mmScaler.fit(resDF.loc[scaledMask, cN].to_numpy().reshape(-1, 1))
        resDF[cN + '_scaled'] = mmScaler.transform(resDF[cN].to_numpy().reshape(-1, 1))
    if arguments['exportToDeepSpine']:
        deepSpineExportPath = os.path.join(
            alignSubFolder,
            prefix + '_{}_{}_export.h5'.format(
                arguments['inputBlockSuffix'], arguments['window']))
        for cN in ['noiseCeil', 'covariance', 'covariance_q_scale']:
            resDF[cN].to_hdf(deepSpineExportPath, cN)
    #
    trialInfo = resDF['noiseCeil'].index.to_frame().reset_index(drop=True)
    dropColumns = []
    dropElectrodes = []
    
    if arguments['plotting']:
        figureOutputFolder = os.path.join(
            figureFolder, arguments['analysisName'])
        if not os.path.exists(figureOutputFolder):
            os.makedirs(figureOutputFolder)
        pdfName = os.path.join(figureOutputFolder, 'noise_ceil_halves.pdf')
        with PdfPages(pdfName) as pdf:
            plotIndex = pd.MultiIndex.from_frame(
                trialInfo.loc[:, [
                    'electrode',
                    'RateInHz', 'nominalCurrent', 'feature']])
            for cN in ['noiseCeil', 'covariance_q_scale', 'mse_q_scale']:
                plotDF = (
                    resDF[cN]
                    .unstack(level='feature')
                    .drop(dropElectrodes, axis='index', level='electrode')
                    .drop(dropColumns, axis='columns'))
                plotDF.index = pd.MultiIndex.from_frame(
                    plotDF.index.to_frame(index=False).loc[
                        :, ['electrode', 'RateInHz', 'nominalCurrent']])
                fig, ax = plt.subplots(figsize=(12, 12))
                sns.heatmap(
                    plotDF, ax=ax,
                    center=0, vmin=-1, vmax=1)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
                ax.set_title(cN)
                plt.tight_layout()
                pdf.savefig()
                if arguments['showPlots']:
                    plt.show()
                else:
                    plt.close()
    keepMask = ((resDF['noiseCeil'] > 0.4) & (resDF['covariance_q_scale'] > 0.1))
    keepFeats = (resDF.loc[keepMask, 'noiseCeil'].index.to_frame().reset_index(drop=True).groupby(['RateInHz', 'nominalCurrent', 'electrode'])['feature'])
    keepFeats.name = 'numFeatures'
    #
    minReliabilityPerEMG = (
        resDF['noiseCeil']
        .unstack(level='feature')
        .drop(dropElectrodes, axis='index', level='electrode')
        .drop(dropColumns, axis='columns').quantile(0.75))
    minReliabilityPerElectrode = (
        resDF['noiseCeil']
        .unstack(level='electrode')
        .drop(dropColumns, axis='index', level='feature')
        .drop(dropElectrodes, axis='columns').quantile(0.75))
    if arguments['plotting']:
        plotCovar = (
            resDF.loc[keepMask, 'covariance_q_scale']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature'))
        plotNoiseCeil = (
            resDF.loc[keepMask, 'noiseCeil']
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature')
            )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.distplot(plotNoiseCeil, ax=ax, kde=False)
        ax.set_title('Noise ceiling histogram')
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Count')
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_histogram.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
        # 
    if arguments['plotting']:
        plotDF = pd.concat(
            {
                'noiseCeil': plotNoiseCeil,
                'covar': plotCovar}, axis='columns').reset_index()
        plotDF['nominalCurrent'] *= -1
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            x='covar', y='noiseCeil',
            hue='electrode', size='nominalCurrent', style='feature',
            markers=EMGStyleMarkers,
            data=plotDF, ax=ax, alpha=0.75, sizes=(10, 100))
        ax.set_xlabel('Scaled Covariance')
        ax.set_ylabel('Reliability')
        ax.set_xlim([-.2, 1])
        # ax.set_ylim([-1, 1])
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_scatterplot.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
    if arguments['plotting']:
        plotDF = (
            resDF.loc[keepMask, :]
            .drop(dropElectrodes, axis='index', level='electrode')
            .drop(dropColumns, axis='index', level='feature')
            .reset_index()
            )
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(x='feature', y='noiseCeil', data=plotDF, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-60)
        fig.savefig(os.path.join(figureOutputFolder, 'noise_ceil_per_emg.pdf'))
        if arguments['showPlots']:
            plt.show()
        else:
            plt.close()
