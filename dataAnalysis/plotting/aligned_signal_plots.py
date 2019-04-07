import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import seaborn as sns

def rasterPlot(*args, **kwargs):
    ax = plt.gca()
    data = kwargs.pop('data')
    yName = kwargs.pop('y')
    xName = kwargs.pop('x')
    uniqueIdx = pd.unique(data[yName])
    idxLookup = {
        uIdx: idx
        for idx, uIdx in enumerate(uniqueIdx)}
    ax.scatter(
        x=data[xName],
        y=data[yName].map(idxLookup),
        **kwargs)
    return


def getRasterFacetIdx(
        plotDF, y, row=None, col=None):
    plotDF.loc[:, y + '_facetIdx'] = np.nan
    dummyG = sns.FacetGrid(
        plotDF,
        row=row, col=col)
    for name, group in dummyG.facet_data():
        uniqueIdx = pd.unique(group[y])
        idxLookup = {
            uIdx: idx
            for idx, uIdx in enumerate(uniqueIdx)}
        plotDF.loc[group.index, y + '_facetIdx'] = (
            group[y].map(idxLookup))
    return plotDF

def twinAxFacetGrid(
        mainFun, mainKws,
        twinFun, twinKws):
    def twinAxPlotFun(*args, **kwargs):
        return
    return twinAxPlotFun
