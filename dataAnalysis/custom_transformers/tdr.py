from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
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


def poisson_pseudoR2(estimator, X, y):
    # adapted from https://github.com/KordingLab/spykesMLs
    # This is our scoring function. Implements pseudo-R2
    #
    # yhat is the prediction
    yhat = estimator.results_.predict(X).to_numpy()
    # y null is the mean of the training data
    ynull = np.mean(estimator.model_.endog)
    yhat = yhat.reshape(y.shape)
    eps = np.spacing(1)
    L1 = np.sum(y*np.log(eps+yhat) - yhat)
    # L1_v = y*np.log(eps+yhat) - yhat
    L0 = np.sum(y*np.log(eps+ynull) - ynull)
    LS = np.sum(y*np.log(eps+y) - y)
    R2 = 1-(LS-L1)/(LS-L0)
    return R2


class SMWrapper(BaseEstimator, RegressorMixin):
    """
        A universal sklearn-style wrapper for statsmodels regressors
        based on https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible/
        by David Dale
    """
    def __init__(
            self, sm_class, sm_kwargs={},
            featureMask=slice(None),
            regAlpha=None, regL1Wt=None, regRefit=True):
        self.sm_class = sm_class
        self.sm_kwargs = sm_kwargs
        self.regAlpha = regAlpha
        self.regL1Wt = regL1Wt
        self.regRefit = regRefit
        self.featureMask = featureMask
        pass

    def fit(self, X, y):
        try:
            self.model_ = self.sm_class(
                y, X.iloc[:, self.featureMask],
                **self.sm_kwargs)
        except Exception:
            pdb.set_trace()
        if self.regAlpha is None:
            self.results_ = self.model_.fit()
        else:
            self.results_ = self.model_.fit_regularized(
                alpha=self.regAlpha, L1_wt=self.regL1Wt,
                refit=self.regRefit)

    def predict(self, X):
        return self.results_.predict(X.iloc[:, self.featureMask])

    def score(self, X, y=None):
        if 'Poisson' in str(self.sm_kwargs['family']):
            return poisson_pseudoR2(self, X.iloc[:, self.featureMask], y)


class SingleNeuronRegression():

    def __init__(
            self,
            featuresDF=None, targetDF=None,
            featureScalers=None, targetScalers=None, addIntercept=True,
            model=None, modelKWargs={},
            cv=None, tTestAlpha=0.01, conditionNames=None,
            plotting=False, verbose=False):
        self.addIntercept = addIntercept
        #
        self.model = model
        self.modelKWargs = modelKWargs
        #
        self.cv = cv
        #
        self.tTestAlpha = tTestAlpha
        #
        self.plotting = plotting
        self.verbose = verbose
        #
        featuresDF = applyScalers(featuresDF, featureScalers)
        targetDF = applyScalers(targetDF, targetScalers)
        #
        if self.addIntercept:
            featuresDF.loc[:, ('intercept', 0)] = 1
        #
        trialInfo = (
            featuresDF
            .index.to_frame()
            .reset_index(drop=True)
            .loc[:, conditionNames + ['bin']])
        for idx, (name, group) in enumerate(trialInfo.groupby(conditionNames)):
            trialInfo.loc[group.index, 'group'] = idx
        prelimSkf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        # pdb.set_trace()
        # WIP: trick to keep trials continuous
        binMask = (trialInfo['bin'] == trialInfo['bin'].iloc[0])
        idxGen = prelimSkf.split(
            trialInfo.iloc[binMask.to_numpy(), 0].to_numpy(),
            trialInfo.loc[binMask.to_numpy(), 'group'].to_numpy())
        skfIndex = [i for i in idxGen]
        idxTest = trialInfo.iloc[binMask.to_numpy(), 0].index[(skfIndex[0][1])]
        idxTrain = trialInfo.iloc[binMask.to_numpy(), 0].index[(np.concatenate([i[1] for i in skfIndex[1:]]))]
        trialInfo['split'] = np.nan
        trialInfo.iloc[idxTrain, -1] = 'train'
        trialInfo.iloc[idxTest, -1] = 'test'
        trialInfo.fillna(method='ffill', inplace=True)
        trainMask = (trialInfo['split'] == 'train').to_numpy()
        testMask = (trialInfo['split'] == 'test').to_numpy()
        self.XTrain = featuresDF.iloc[trainMask, :]
        self.XTrain.columns = self.XTrain.columns.to_list()
        self.yTrain = targetDF.iloc[trainMask, :]
        # self.yTrain.columns = self.yTrain.columns.to_list()
        self.XTest = featuresDF.iloc[testMask, :]
        self.XTest.columns = self.XTest.columns.to_list()
        self.yTest = targetDF.iloc[testMask, :]
        # self.yTest.columns = self.yTest.columns.to_list()
        # pdb.set_trace()
        self.betas = pd.DataFrame(
            0, index=targetDF.columns, columns=featuresDF.columns)
        self.pvals = pd.DataFrame(
            np.nan, index=self.betas.index,
            columns=self.betas.columns)
        self.regressionList = {k: {} for k in targetDF.columns}
        for colName in self.yTrain:
            reg = self.model(**self.modelKWargs)
            self.regressionList[colName].update({'reg': reg})
        pass

    def cross_val_score(self):
        #  fit the regression models
        for colName in self.yTrain:
            y = self.yTrain.loc[:, colName]
            scores = cross_val_score(
                self.regressionList[colName]['reg'],
                self.XTrain, y, cv=self.cv - 1)
            if self.verbose:
                print('{}: mean score {}, std {}'.format(
                    colName, np.mean(scores), np.std(scores)
                    ))
            self.regressionList[colName].update({
                'scoresCV': scores
                })
        return

    def apply_gridSearchCV(self, gridParams):
        #  fit the regression models
        for colName in self.yTrain:
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            gs = GridSearchCV(reg, gridParams, refit=False, cv=self.cv - 1)
            gs.fit(self.XTrain, y)
            bestIndex = gs.best_index_
            reg.set_params(**gs.best_params_)
            self.regressionList[colName].update({
                'gridSearchCV': gs,
                'mean_test_score': gs.cv_results_['mean_test_score'][bestIndex],
                'std_test_score': gs.cv_results_['std_test_score'][bestIndex],
                })
            if self.verbose:
                print("Best parameters set found:\n{}".format(
                    gs.best_params_))
        return

    def fit(self):
        #  fit the regression models
        for colName in self.yTrain:
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            reg.fit(self.XTrain, y)
            if self.verbose and hasattr(reg, 'results_'):
                if hasattr(reg.results_, 'summary'):
                    print(reg.results_.summary())
                    print('params \n')
                    print(reg.results_.params)
            if hasattr(reg, 'results_'):
                self.betas.loc[colName, self.betas.columns[reg.featureMask]] = reg.results_.params
                if hasattr(reg.results_, 'pvals'):
                    self.pvals.loc[colName, self.pvals.columns[reg.featureMask]] = reg.results_.pvalues
        #
        self.significantBetas = None
        if hasattr(reg, 'results_'):
            if hasattr(reg.results_, 'pvals'):
                origShape = self.pvals.shape
                flatPvals = self.pvals.to_numpy().reshape(-1)
                try:
                    _, fixedPvals, _, _ = mt(flatPvals, method='holm')
                except Exception:
                    fixedPvals = flatPvals * flatPvals.size
                self.pvals.iloc[:, :] = fixedPvals.reshape(origShape)
                self.significantBetas = self.pvals < self.tTestAlpha
        if self.significantBetas is None:
            self.significantBetas = self.betas > 0
        return self

    def clear_data(self):
        del self.XTrain, self.yTrain, self.XTest, self.yTest
        return

    def plot_xy(self):
        scores = [
            {
                'unit': k, 'score': v['mean_test_score'],
                'std_score': v['std_test_score']}
            for k, v in self.regressionList.items()]
        scoresDF = pd.DataFrame(scores)
        unitName = scoresDF.loc[scoresDF['score'].idxmax(), 'unit']
        thisReg = self.regressionList[unitName]
        prediction = thisReg['reg'].predict(self.XTest).to_numpy()
        y = self.yTest[unitName].to_numpy()
        #
        if True:
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(y / max(y), label='original')
            ax[0].plot(prediction, label='prediction')
            ax[0].set_title('{}: pR^2 = {:.2f}'.format(
                unitName,
                scoresDF.loc[scoresDF['score'].idxmax(), 'score']))
            ax[0].set_xlabel('samples')
            ax[0].set_ylabel('normalized (spk/s)')
        for idx, beta in enumerate(thisReg['reg'].results_.params):
            dfIdx = thisReg['reg'].featureMask[idx]
            x = beta * self.XTest.iloc[:, dfIdx].to_numpy()
            if thisReg['reg'].results_.pvalues[idx] < self.tTestAlpha:
                ax[1].plot(
                    x,
                    label='{}'.format(self.XTest.iloc[:, dfIdx].name))
            else:
                ax[2].plot(
                    x, ls='--',
                    label='{}'.format(self.XTest.iloc[:, dfIdx].name))
        ax[1].set_title('p < {} regressors'.format(self.tTestAlpha))
        ax[2].set_title('p > {} regressors'.format(self.tTestAlpha))
        for thisAx in ax:
            thisAx.legend()
        plt.show()
        return


class TargetedDimensionalityReduction(TransformerMixin):

    def __init__(
            self,
            featuresDF=None, targetDF=None,
            featureScalers=None, targetScalers=None, addIntercept=True,
            model=None, modelKWargs={}, cv=None,
            tTestAlpha=0.01,
            nPCAComponents=None, conditionNames=None,
            enableTDR=True,
            plotting=False, verbose=False):
        #
        self.plotting = plotting
        self.verbose = verbose
        self.conditionNames = conditionNames
        self.regressorNames = featuresDF.columns.copy()
        #
        self.regression_ = SingleNeuronRegression(
            featuresDF=featuresDF, targetDF=targetDF,
            featureScalers=featureScalers,
            targetScalers=targetScalers, addIntercept=addIntercept,
            model=model, modelKWargs=modelKWargs, cv=cv,
            tTestAlpha=tTestAlpha,
            plotting=plotting, verbose=verbose)
        #
        self.betaMax = None
        if nPCAComponents is not None:
            self.pca = PCA(n_components=nPCAComponents)
        else:
            self.pca = None
        pass

    def fit(self, X=None, y=None):
        self.regression_.fit()
        dropColumns = self.regression_.significantBetas.columns[~self.regression_significantBetas.any()]
        # self.betas.drop(columns=dropColumns, inplace=True)
        self.regressorNames = (
            self.regressorNames
            .drop(dropColumns)
            .unique(level='feature'))
        # calculate PCA of neural state
        if self.pca is not None:
            conditionAverages = self.targetDF.groupby(self.conditionNames).agg('mean')
            self.pca.fit(conditionAverages.to_numpy())
        # manipulate betas to get dimensionality reduction
        transposedBetas = (
            self.betas
            .drop(columns=dropColumns)
            .unstack(level='positionBin').transpose())
        if self.pca is not None:
            denoisedBetas = pd.DataFrame(
                self.pca.inverse_transform(self.pca.transform(transposedBetas)),
                index=transposedBetas.index,
                columns=transposedBetas.columns)
        else:
            denoisedBetas = transposedBetas
        maxBins = []
        for name, group in denoisedBetas.groupby('feature'):
            maxBins.append((group ** 2).sum(axis='columns').idxmax())
        self.betaMax = denoisedBetas.loc[maxBins, :].transpose()
        #
        if self.enableTDR:
            self.q, r = np.linalg.qr(self.betaMax)
        else:
            self.q = None
        return self

    def calculate_tdr_axes(self):
        if self.betaMax is None:
            self.fit()
        self.q, r = np.linalg.qr(self.betaMax)
        return

    def transform(self, X):
        if self.q is not None:
            return np.dot(self.q.transpose(), X.transpose()).transpose()
        else:
            raise(Exception('Must run calculate_tdr_axes first!'))

    def clear_data(self):
        del self.featuresDF, self.targetDF
        return
