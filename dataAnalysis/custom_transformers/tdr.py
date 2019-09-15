from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests as mt
import pdb, traceback
import statsmodels


class TargetedDimensionalityReduction(TransformerMixin):
    def __init__(
            self,
            featuresDF=None, targetDF=None,
            model=None, modelKWargs={},
            featureScalers=None, targetScalers=None,
            addLags=None, decimate=1, rollingWindow=None,
            timeAxisName=None,
            nPCAComponents=None, conditionNames=None,
            addIntercept=True,
            plotting=False, verbose=False):
        #
        self.modelKWargs = modelKWargs
        self.model = model
        self.verbose = verbose
        self.timeAxisName = timeAxisName
        self.conditionNames = conditionNames
        self.addIntercept = addIntercept
        #
        def applyScalers(DF, listOfScalers):
            for scaler, listOfColumns in listOfScalers:
                try:
                    featuresMatchMaskAll = (
                        DF
                        .columns
                        .get_level_values('feature')
                        .isin(listOfColumns))
                    trainIndices = []
                    for colName in listOfColumns:
                        featuresMatchMask = (
                            DF
                            .columns
                            .get_level_values('feature')
                            .isin([colName]))
                        firstLag = (
                            DF.columns
                            .get_level_values('lag')[featuresMatchMask][0])
                        trainIndices.append((colName, firstLag))
                    scaler.fit(
                        DF.loc[:, trainIndices]
                        .to_numpy().reshape(-1, 1))
                    originalShape = DF.iloc[:, featuresMatchMaskAll].shape
                    scaledVal = scaler.transform(
                        (
                            DF
                            .iloc[:, featuresMatchMaskAll]
                            .to_numpy().reshape(-1, 1))
                        ).reshape(originalShape)
                    DF.iloc[:, featuresMatchMaskAll] = scaledVal
                except Exception:
                    traceback.print_exc()
                    pdb.set_trace()
            return DF
        #
        dropIndex = featuresDF.index[featuresDF.isna().T.any()]
        targetDF.drop(index=dropIndex, inplace=True)
        featuresDF.drop(index=dropIndex, inplace=True)
        #
        featuresDF = applyScalers(featuresDF, featureScalers)
        targetDF = applyScalers(targetDF, targetScalers)
        # finally, decimate if need to
        self.targetDF = targetDF.iloc[::decimate, :]
        self.featuresDF = featuresDF.iloc[::decimate, :]
        #
        if self.addIntercept:
            self.featuresDF.loc[:, ('intercept', 0)] = 1
        self.metaDF = self.featuresDF.index.to_frame().reset_index(drop=True)
        uniqueBins = np.unique(self.metaDF[timeAxisName])
        uniqueUnits = targetDF.columns.to_numpy()
        betaIndex = pd.MultiIndex.from_product(
            [uniqueUnits, uniqueBins],
            names=['unit', timeAxisName])
        self.regressorNames = self.featuresDF.columns
        self.betas = pd.DataFrame(
            np.nan, index=betaIndex, columns=self.featuresDF.columns)
        self.betaMax = None
        self.regressionList = []
        self.pca = PCA(n_components=nPCAComponents)
        pass

    def fit(self, X=None, y=None):
        for binName, yIndex in self.metaDF.groupby(self.timeAxisName):
            groupMask = self.metaDF.index.isin(yIndex.index)
            pvals = pd.DataFrame(
                np.nan, index=self.betas.index,
                columns=self.betas.columns)
            for colName in self.targetDF:
                x = self.featuresDF.loc[groupMask, :]
                x.columns = x.columns.to_list()
                y = self.targetDF.loc[groupMask, colName]
                # pdb.set_trace()
                reg = self.model(y, x, **self.modelKWargs)
                regResults = reg.fit()
                if self.verbose:
                    print(regResults.summary())
                try:
                    # OLS models don't have llf or pseudor2
                    # it's fine if this fails
                    pr2 = 1 - (regResults.llf / regResults.llnull)
                    if self.verbose:
                        print('McFadden pseudo-R-squared: {})'.format(pr2))
                except Exception:
                    traceback.print_exc()
                self.betas.loc[(colName, binName), :] = regResults.params
                pvals.loc[(colName, binName), :] = regResults.pvalues
                thisResult = {
                    self.timeAxisName: binName,
                    'unit': colName, 'reg': regResults
                    }
                try:
                    # OLS models don't have llf or pseudor2
                    # it's fine if this fails
                    thisResult.update({'pseudorsquared': pr2})
                except Exception:
                    traceback.print_exc()
                self.regressionList.append(thisResult)
        self.betas.dropna(inplace=True)
        origShape = pvals.shape
        flatPvals = pvals.to_numpy().reshape(-1)
        try:
            _, fixedPvals, _, _ = mt(flatPvals, method='holm')
        except Exception:
            fixedPvals = flatPvals * flatPvals.size
        pvals.iloc[:, :] = fixedPvals.reshape(origShape)
        alpha = 0.01
        significantBetas = pvals < alpha
        dropColumns = significantBetas.columns[~significantBetas.any()]
        # self.betas.drop(columns=dropColumns, inplace=True)
        self.regressorNames = (
            self.regressorNames
            .drop(dropColumns)
            .unique(level='feature'))
        conditionAverages = self.targetDF.groupby(self.conditionNames).agg('mean')
        self.pca.fit(conditionAverages.to_numpy())
        transposedBetas = (
            self.betas
            .drop(columns=dropColumns)
            .unstack(level='positionBin').transpose())
        denoisedBetas = pd.DataFrame(
            self.pca.inverse_transform(self.pca.transform(transposedBetas)),
            index=transposedBetas.index,
            columns=transposedBetas.columns)
        maxBins = []
        for name, group in denoisedBetas.groupby('feature'):
            maxBins.append((group ** 2).sum(axis='columns').idxmax())
        self.betaMax = denoisedBetas.loc[maxBins, :].transpose()
        self.q, r = np.linalg.qr(self.betaMax)
        return self

    def transform(self, X):
        return np.dot(self.q.transpose(), X.transpose()).transpose()

    def clear_data(self):
        del self.featuresDF, self.targetDF
        return
