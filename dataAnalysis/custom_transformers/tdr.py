from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests as mt
import pdb, traceback
import statsmodels

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")


class SMWrapper(BaseEstimator, RegressorMixin):
    """
        A universal sklearn-style wrapper for statsmodels regressors
        based on https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible/
        by David Dale
    """
    def __init__(
            self, model_class, model_kwargs={},
            regAlpha=None, regL1Wt=None, regRefit=True):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.regAlpha = regAlpha
        self.regL1Wt = regL1Wt
        self.regRefit = regRefit

    def fit(self, X, y):
        self.model_ = self.model_class(y, X, **self.model_kwargs)
        if self.regAlpha is None:
            self.results_ = self.model_.fit()
        else:
            self.results_ = self.model_.fit_regularized(
                alpha=self.regAlpha, L1_wt=self.regL1Wt,
                refit=self.regRefit)

    def predict(self, X):
        return self.results_.predict(X)


class TargetedDimensionalityReduction(TransformerMixin):

    def __init__(
            self,
            featuresDF=None, targetDF=None,
            model=None, modelKWargs={},
            regAlpha=None, regL1Wt=None, nCV=None,
            featureScalers=None, targetScalers=None,
            addLags=None, decimate=1, rollingWindow=None,
            timeAxisName=None, tTestAlpha=0.01,
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
        self.tTestAlpha = tTestAlpha
        self.regAlpha = regAlpha
        self.regL1Wt = regL1Wt
        self.nCV = nCV
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
                if self.regAlpha is None:
                    regResults = reg.fit()
                else:
                    regResults = reg.fit_regularized(
                        alpha=self.regAlpha, L1_wt=self.regL1Wt, refit=True)

                if self.verbose:
                    print(regResults.summary())
                try:
                    # OLS models don't have llf or pseudor2
                    # it's fine if this fails
                    #
                    # Statsmodels suggests this formula
                    # pr2 = 1 - (regResults.llf / regResults.llnull)
                    # 
                    # Benjamin et al 2018 suggests this formula
                    pr2 = 1 - (regResults.deviance / regResults.null_deviance)
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
        significantBetas = pvals < self.tTestAlpha
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

    def plot_xy(self):
        rsquared = pd.DataFrame([{'unit': i['unit'], 'rsquared': i['pseudorsquared']} for i in self.regressionList])
        unitName = rsquared.loc[rsquared['rsquared'].idxmax(), 'unit']
        regressionEntry = [i for i in self.regressionList if i['unit'] == unitName][0]
        #
        thisReg = regressionEntry['reg']
        prediction = thisReg.predict()
        y = self.targetDF[unitName].to_numpy()
        #
        if True:
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(y / max(y), label='original')
            ax[0].plot(prediction, label='prediction')
            ax[0].set_title('{}: pR^2 = {}'.format(unitName, rsquared.loc[rsquared['rsquared'].idxmax(), 'rsquared']))
            ax[0].set_xlabel('samples')
            ax[0].set_ylabel('normalized (spk/s)')
        for idx, beta in enumerate(thisReg.params):
            x = beta * self.featuresDF.iloc[:, idx].to_numpy()
            if thisReg.pvalues[idx] < self.tTestAlpha:
                ax[1].plot(x, label='{}'.format(self.featuresDF.iloc[:, idx].name))
            else:
                ax[2].plot(x, ls='--', label='{}'.format(self.featuresDF.iloc[:, idx].name))
        ax[1].set_title('p < {} regressors'.format(self.tTestAlpha))
        ax[2].set_title('p > {} regressors'.format(self.tTestAlpha))
        for thisAx in ax:
            thisAx.legend()
        plt.show()
        return
