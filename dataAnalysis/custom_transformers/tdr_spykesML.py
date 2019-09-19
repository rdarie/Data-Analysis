from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests as mt
import pdb, traceback
import statsmodels

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")


class TargetedDimensionalityReduction(TransformerMixin):
    def __init__(
            self,
            featuresDF=None, targetDF=None,
            model=None, modelKWargs={}, nCV=None,
            featureScalers=None, targetScalers=None,
            addLags=None, decimate=1, rollingWindow=None,
            timeAxisName=None, alpha=0.01,
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
        self.alpha = alpha
        self.nCV = nCV
        #
        if (model.tunemodel == 'glm'):
            model.set_params({'cv': nCV})
        model.set_params(modelKWargs)
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
            for idx, colName in enumerate(self.targetDF):
                print('Fitting neuron {}...'.format(colName))
                x = self.featuresDF.loc[groupMask, :].to_numpy()
                y = self.targetDF.loc[groupMask, colName].to_numpy()
                self.model.fit(x, y)
                if (idx == 0) and (self.model.tunemodel == 'glm'):
                    self.model.set_params(
                        {'reg_lambda': [self.model.model.reg_lambda]})
                #
                self.betas.loc[(colName, binName), :] = self.model.model.beta_
                thisResult = {
                    self.timeAxisName: binName,
                    'unit': colName, 'reg': self.model.model.copy()
                    }
                try:
                    # OLS models don't have llf or pseudor2
                    # it's fine if this fails
                    pr2 = max(self.model.GLMCV.scores_)
                    print('pseudoR2 = {:.4f}'.format(pr2))
                    thisResult.update({'pseudorsquared': pr2})
                except Exception:
                    traceback.print_exc()
                self.regressionList.append(thisResult)
        # self.betas.dropna(inplace=True)
        significantBetas = self.betas.notna()
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
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(y / max(y), label='original')
            ax[0].plot(prediction, label='prediction')
            ax[0].set_title('{}: pR^2 = {}'.format(unitName, rsquared.loc[rsquared['rsquared'].idxmax(), 'rsquared']))
            ax[0].set_xlabel('samples')
            ax[0].set_ylabel('normalized (spk/s)')
        for idx, beta in enumerate(thisReg.params):
            x = beta * self.featuresDF.iloc[:, idx].to_numpy()
            if thisReg.pvalues[idx] < self.alpha:
                ax[1].plot(x, label='{}'.format(self.featuresDF.iloc[:, idx].name))
            else:
                ax[2].plot(x, ls='--', label='{}'.format(self.featuresDF.iloc[:, idx].name))
        ax[1].set_title('p < {} regressors'.format(self.alpha))
        ax[2].set_title('p > {} regressors'.format(self.alpha))
        for thisAx in ax:
            thisAx.legend()
        plt.show()
        return
