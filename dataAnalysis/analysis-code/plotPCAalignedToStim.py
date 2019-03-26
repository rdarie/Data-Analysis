import os, pdb

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from currentExperiment import *

featurePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_features.hdf')
unpackedFeatures = pd.read_hdf(featurePath, 'features')
unpackedFeatures = unpackedFeatures.query('(bin > -0.125) & (bin < 0.25)')

# pdb.set_trace()
#  unpackedFeatures.loc[
#      unpackedFeatures['ampCat'] == 'Control',
#      'moveCat'] = ('Control')

#  plotting orders and other options
sns.set_style("whitegrid")
#  movementOrder = [-1, 1, 0]
moveCatOrder = [
    'outbound', 'reachedPeak',
    'return', 'reachedBase', 'Control']
#  pdb.set_trace()

dataQuery = '&'.join([
    '(RateInHz >= 100)',
    '((ampCat == \'M\') | (ampCat == \'Control\'))',
    ])
plotDF = unpackedFeatures.query(dataQuery)

#  plotDF = unpackedFeatures
#  facet_kws={'despine': True, 'ylim': (-25, 200)}
g = sns.relplot(
    row='electrode', col='moveCat', col_order=moveCatOrder,
    x='bin', y='PC1', hue='amplitude',
    data=plotDF, kind='line', facet_kws={'despine': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'PC1.pdf'))
plt.close()

g = sns.relplot(
    row='electrode', col='moveCat', col_order=moveCatOrder,
    x='bin', y='PC2', hue='amplitude',
    data=plotDF, kind='line', facet_kws={'despine': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'PC2.pdf'))
plt.close()

g = sns.relplot(
    row='electrode', col='moveCat', col_order=moveCatOrder,
    x='PC1', y='PC2', hue='amplitude',
    data=plotDF, kind='scatter', facet_kws={'despine': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'PC1vs2.pdf'))
plt.close()

#  dataQuery = '&'.join([
#      '((ampCat == \'M\') | (ampCat == \'Control\'))',
#      ])
dataQuery = '&'.join([
    '(RateInHz >= 100)',
    '((ampCat == \'M\') | (ampCat == \'Control\'))',
    ])
plotDF = unpackedFeatures.query(dataQuery)
#  plotDF = unpackedFeatures
g = sns.relplot(
    col='moveCat', col_order=moveCatOrder,
    x='bin', y='position',
    data=plotDF, kind='line',
    facet_kws={'despine': True, 'sharey': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'position.pdf'))
plt.close()

g = sns.relplot(
    col='moveCat', col_order=moveCatOrder,
    x='bin', y='ins_td0', hue='amplitude',
    data=plotDF, kind='line',
    facet_kws={'despine': True, 'sharey': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'stimArtifact.pdf'))
plt.close()

g = sns.relplot(
    col='moveCat', col_order=moveCatOrder,
    x='bin', y='tdAmplitude', hue='amplitude',
    data=plotDF, kind='line',
    facet_kws={'despine': True, 'sharey': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'stimTrace.pdf'))
plt.close()

dataQuery = '&'.join([
    '(amplitude == 0)',
    ])
plotDF = unpackedFeatures.query(dataQuery)
#  plotDF = unpackedFeatures
g = sns.relplot(
    col='moveCat', col_order=moveCatOrder,
    x='bin', y='ins_td0', hue='ampCat',
    data=plotDF, kind='line',
    facet_kws={'despine': True, 'sharey': True})
plt.show(block=False)
plt.pause(3)
plt.savefig(os.path.join(figureFolder, 'ins_td0.pdf'))
plt.close()
