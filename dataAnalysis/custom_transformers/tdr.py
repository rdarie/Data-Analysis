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
        for scaler, listOfColumns in featureScalers:
            try:
                originalShape = featuresDF.loc[:, listOfColumns].shape
                scaledVal = (
                    scaler
                    .fit_transform(
                        featuresDF.loc[:, listOfColumns]
                        .to_numpy().reshape(-1, 1))
                    .reshape(originalShape))
                featuresDF.loc[:, listOfColumns] = scaledVal
            except Exception:
                traceback.print_exc()
                pdb.set_trace()
        #
        for scaler, listOfColumns in targetScalers:
            try:
                originalShape = targetDF.loc[:, listOfColumns].shape
                scaledVal = (
                    scaler
                    .fit_transform(
                        targetDF.loc[:, listOfColumns]
                        .to_numpy().reshape(-1, 1))
                    .reshape(originalShape))
                targetDF.loc[:, listOfColumns] = scaledVal
            except Exception:
                traceback.print_exc()
                pdb.set_trace()
        #
        extendedColumns = []
        for key in sorted(featuresDF.columns):
            extendedColumns += [(key, valueItem) for valueItem in addLags[key]]
        extendedColumnIndex = pd.MultiIndex.from_tuples(
            sorted(extendedColumns), names=['taskVariable', 'lag'])
        # trick to avoid indexing nonsense: remove index and add back at the end
        extendedFeaturesDF = pd.DataFrame(
            np.nan,
            index=range(len(featuresDF.index)),
            columns=extendedColumnIndex)
        targetDF.reset_index(drop=True, inplace=True)
        for name, group in featuresDF.reset_index().groupby(['segment', 't']):
            targetDF.loc[group.index, :] = (
                targetDF.loc[group.index, :]
                .rolling(rollingWindow, center=True)
                .mean())
            for feature, lag in extendedColumns:
                if isinstance(lag, int):
                    shiftedFeature = (
                        group[feature]
                        .shift(lag))
                    if rollingWindow is not None:
                        shiftedFeature = (
                            shiftedFeature
                            .rolling(rollingWindow, center=True)
                            .mean())
                    extendedFeaturesDF.loc[
                        group.index, (feature, lag)] = shiftedFeature
                if isinstance(lag, tuple):
                    shiftedFeature = (
                        group[feature]
                        .shift(lag[0])
                        .rolling(lag[1], center=True)
                        .mean())
                    extendedFeaturesDF.loc[
                        group.index, (feature, lag)] = shiftedFeature
        extendedFeaturesDF.drop(
            columns=(
                extendedFeaturesDF
                .columns[extendedFeaturesDF.nunique() == 1]),
            inplace=True)
        # indexes must be equal (interchangeable)!
        targetDF.index = featuresDF.index
        extendedFeaturesDF.index = featuresDF.index
        dropIndex = extendedFeaturesDF.index[extendedFeaturesDF.isna().T.any()]
        extendedFeaturesDF.drop(index=dropIndex, inplace=True)
        targetDF.drop(index=dropIndex, inplace=True)
        featuresDF.drop(index=dropIndex, inplace=True)
        # finally, decimate if need to
        self.targetDF = targetDF.iloc[::decimate, :]
        self.featuresDF = extendedFeaturesDF.iloc[::decimate, :]
        #
        if self.addIntercept:
            pdb.set_trace()
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
            .unique(level='taskVariable'))
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
        for name, group in denoisedBetas.groupby('taskVariable'):
            maxBins.append((group ** 2).sum(axis='columns').idxmax())
        self.betaMax = denoisedBetas.loc[maxBins, :].transpose()
        self.q, r = np.linalg.qr(self.betaMax)
        return self

    def transform(self, X):
        return np.dot(self.q.transpose(), X.transpose()).transpose()

    def clear_data(self):
        del self.featuresDF, self.targetDF
        return
