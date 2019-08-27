from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import statsmodels.api as sm


class TargetedDimensionalityReduction(TransformerMixin):
    def __init__(
            self, timeAxisName=None,
            nPCAComponents=None, conditionNames=None,
            featuresDF=None, targetDF=None, plotting=False):
        self.timeAxisName = timeAxisName
        self.conditionNames = conditionNames
        self.featuresDF = featuresDF
        self.targetDF = targetDF
        self.metaDF = featuresDF.index.to_frame().reset_index(drop=True)
        uniqueBins = np.unique(self.metaDF[timeAxisName])
        uniqueUnits = targetDF.columns.to_numpy()
        betaIndex = pd.MultiIndex.from_product(
            [uniqueUnits, uniqueBins],
            names=['unit', timeAxisName])
        self.regressorNames = np.concatenate(
            [
                ['intercept'],
                featuresDF.columns.to_numpy()])
        self.betas = pd.DataFrame(
            np.nan, index=betaIndex, columns=self.regressorNames)
        self.regressionList = []
        self.pca = PCA(n_components=nPCAComponents)
        pass

    def fit(self, X=None, y=None):
        for binName, yIndex in self.metaDF.groupby(self.timeAxisName):
            groupMask = self.metaDF.index.isin(yIndex.index)
            x = self.featuresDF.loc[groupMask, :].to_numpy()
            x2 = sm.add_constant(x, has_constant='add')
            for colName in self.targetDF:
                y = self.targetDF.loc[groupMask, colName].to_numpy()
                reg = sm.OLS(y, x2).fit()
                self.betas.loc[(colName, binName), :] = reg.params
                self.regressionList.append(
                    {
                        self.timeAxisName: binName,
                        'unit': colName, 'reg': reg
                        }
                )
            self.betas.dropna(inplace=True)
            self.betas.columns.name = 'taskVariable'
        conditionAverages = self.targetDF.groupby(self.conditionNames).agg('mean')
        self.pca.fit(conditionAverages.to_numpy())
        transposedBetas = self.betas.unstack(level='positionBin').transpose()
        denoisedBetas = pd.DataFrame(
            self.pca.inverse_transform(self.pca.transform(transposedBetas)),
            index=transposedBetas.index,
            columns=transposedBetas.columns)
        maxBins = []
        for name, group in denoisedBetas.groupby('taskVariable'):
            maxBins.append((group ** 2).sum(axis='columns').idxmax())
        betaMax = denoisedBetas.loc[maxBins, :]
        self.q, r = np.linalg.qr(betaMax.transpose())
        return self

    def transform(self, X):
        return np.dot(self.q.transpose(), X.transpose()).transpose()
