from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    GridSearchCV, StratifiedKFold)
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests as mt
import pdb, traceback
import os
import scipy.optimize
from itertools import product
import joblib as jb
import statsmodels
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices)
import pyglmnet
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")


def crossValidationScores(
        X, y, estimator,
        estimatorKWArgs={},
        crossvalKWArgs={},
        joblibBackendArgs={},
        verbose=0):
    with jb.parallel_backend(**joblibBackendArgs):
        instance = estimator(**estimatorKWArgs)
        scores = cross_validate(instance, X, y, verbose=verbose, **crossvalKWArgs)
    # train on all of the "working" samples, eval on the "validation"
    if hasattr(crossvalKWArgs['cv'], 'work'):
        workingIdx = crossvalKWArgs['cv'].work
        workEstim = estimator(**estimatorKWArgs)
        workX = X.iloc[workingIdx, :]
        workY = y.iloc[workingIdx]
        validIdx = crossvalKWArgs['cv'].validation
        valX = X.iloc[validIdx, :]
        valY = y.iloc[validIdx]
        #
        workEstim.fit(workX, workY)
        #
        scores['fit_time'] = np.append(scores['fit_time'], np.nan)
        scores['score_time'] = np.append(scores['score_time'], np.nan)
        scores['estimator'] = np.append(scores['estimator'], workEstim)
        scores['train_score'] = np.append(scores['train_score'], workEstim.score(workX, workY))
        scores['test_score'] = np.append(scores['test_score'], workEstim.score(valX, valY))
    return scores


def gridSearchHyperparameters(
        X, y, estimator,
        estimatorKWArgs={},
        gridSearchKWArgs={},
        crossvalKWArgs={},
        joblibBackendArgs={},
        verbose=0):
    with jb.parallel_backend(**joblibBackendArgs):
        estimatorProto = estimator(**estimatorKWArgs)
        if isinstance(estimatorProto, ElasticNet):
            gridSearcher = ElasticNetCV(verbose=verbose, **gridSearchKWArgs)
        else:
            gridSearcher = GridSearchCV(
                estimatorProto, verbose=verbose, **gridSearchKWArgs)
        if verbose > 0:
            print('Fitting gridSearchCV...')
        gridSearcher.fit(X, y)
        if verbose:
            print('    Done fitting gridSearchCV!')
        if isinstance(estimatorProto, ElasticNet):
            optParams = {'alpha': gridSearcher.alpha_, 'l1_ratio': gridSearcher.l1_ratio_}
        else:
            optParams = gridSearcher.best_params_
        scoringEstimatorParams = estimatorKWArgs.copy()
        scoringEstimatorParams.update(optParams)
        if verbose:
            print('cross val scoring')
        scores = crossValidationScores(
            X, y, estimator,
            estimatorKWArgs=scoringEstimatorParams,
            crossvalKWArgs=crossvalKWArgs,
            joblibBackendArgs=joblibBackendArgs,
            verbose=verbose)
        if verbose:
            print('    Done cross val scoring!')
    return scores, gridSearcher


def timeShift(x, lag):
    return (
        x.shift(lag)
        .fillna(method='ffill', axis=0)
        .fillna(method='bfill', axis=0)
        )


def shiftSmooth(x, lag=0, winWidth=1):
    halfRollingWin = int(np.ceil(winWidth/2))
    return (
        x.rolling(winWidth, center=True, win_type='gaussian')
        .mean(std=halfRollingWin)
        .shift(lag)
        .fillna(method='ffill', axis=0)
        .fillna(method='bfill', axis=0)
        )


def shiftSmoothDecimate(x, lag=0, winWidth=1, decimate=1):
    halfRollingWin = int(np.ceil(winWidth/2))
    procDF = shiftSmooth(x, lag=lag, winWidth=winWidth)
    if winWidth == 1:
        return procDF.iloc[::decimate]
    else:
        return procDF.iloc[:, halfRollingWin:-halfRollingWin:decimate]


def applyScalersGrouped(DF, listOfScalers):
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
    return DF


def applyScalersGroupedNoLag(DF, listOfScalers):
    for scaler, listOfColumns in listOfScalers:
        featuresMatchMask = (
            DF
            .columns
            .isin(listOfColumns))
        originalShape = DF.iloc[:, featuresMatchMask].shape
        scaledVal = scaler.fit_transform(
            DF.loc[:, featuresMatchMask]
            .to_numpy().reshape(-1, 1)).reshape(originalShape)
        DF.iloc[:, featuresMatchMask] = scaledVal
    return DF


def raisedCos(x, c, dc):
    argCos = (x - c) * np.pi / dc / 2
    argCos[argCos > np.pi] = np.pi
    argCos[argCos < - np.pi] = - np.pi
    return (np.cos(argCos) + 1) / 2


def raisedCosBoundary(
        b=None, DT=None, minX=None, nb=None, plotting=False):
    eps = 1e-20
    nlin = lambda x: np.log(x + eps)
    # fun = lambda c0: (nb - 1) * nlin(c0 + b) + 2 * nlin(c0 + DT + b) - (nb - 1) * nlin(minX + b)
    def fun(c0):
        return raisedCos(
            np.asarray([nlin(minX + b)]),
            np.asarray([nlin(c0 + b)]),
            np.asarray([nlin(c0 + DT + b) - nlin(c0 + b)]) / (nb - 1)
            )[0]
    if plotting:
        plotC0 = np.linspace(-minX, minX * 20, 1000)
        plt.plot(plotC0, fun(plotC0))
        plt.title('solving for cos basis')
        plt.show()
    return scipy.optimize.root(fun, minX * 2).x


def makeRaisedCosBasis(
        nb=None, spacing=None, dt=None, endpoints=None, normalize=False):
    """
        Make nonlinearly stretched basis consisting of raised cosines
    """
    if spacing is None:
        spacing = (endpoints[1] - endpoints[0]) / (nb - 1)
    else:
        assert (nb is None)
    # centers for basis vectors
    ctrs = np.round(np.arange(
        endpoints[0], endpoints[1] + spacing, spacing), decimals=3)
    if nb is None:
        nb = len(ctrs)
    iht = np.arange(
        endpoints[0] - 2 * spacing,
        endpoints[1] + 2 * spacing + dt, dt)
    repIht = np.vstack([iht for i in range(nb)]).transpose()
    nt = iht.size
    repCtrs = np.vstack([ctrs for i in range(nt)])
    ihbasis = raisedCos(repIht, repCtrs, spacing)
    if normalize:
        for colIdx in range(ihbasis.shape[1]):
            ihbasis[:, colIdx] = ihbasis[:, colIdx] / np.sum(ihbasis[:, colIdx])
    return pd.DataFrame(ihbasis, index=iht, columns=ctrs)


def makeLogRaisedCosBasis(
        nb, dt, endpoints, b=0.01, zflag=False, normalize=False):
    """
        Make nonlinearly stretched basis consisting of raised cosines
        Inputs:  nb = # of basis vectors
                 dt = time bin separation for representing basis
                 endpoints = 2-vector containg [1st_peak  last_peak], the peak 
                         (i.e. center) of the last raised cosine basis vectors
                 b = offset for nonlinear stretching of x axis:  y = log(x+b) 
                     (larger b -> more nearly linear stretching)
                 zflag = flag for making (if = 1) finest-timescale basis
                         vector constant below its peak
        
         Outputs:  iht = time lattice on which basis is defined
                   ihbas = orthogonalized basis
                   ihbasis = basis itself
        
         Example call
         iht, ihbas, ihbasis = makeRaisedCosBasis(10, .01, [0, 10], .1);
    """
    #
    eps = 1e-20
    nlin = lambda x: np.log(x + eps)
    invnl = lambda x: np.exp(x) - eps
    assert b > 0
    if isinstance(endpoints, list):
        endpoints = np.array(endpoints)
    #
    yrnge = nlin(endpoints + b)
    db = np.diff(yrnge)/(nb-1)  # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)  # centers for basis vectors
    if ctrs.size > nb:
        ctrs = ctrs[:-1]
    mxt = invnl(yrnge[1]+2*db) - b  # maximum time bin
    iht = np.arange(0, mxt, dt)
    nt = iht.size
    #
    repIht = np.vstack([nlin(iht+b) for i in range(nb)]).transpose()
    repCtrs = np.vstack([ctrs for i in range(nt)])
    ihbasis = raisedCos(repIht, repCtrs, db)
    if zflag:
        tMask = iht < endpoints[0]
        ihbasis[tMask, 0] = 1
    orthobas = scipy.linalg.orth(ihbasis)
    if normalize:
        for colIdx in range(ihbasis.shape[1]):
            ihbasis[:, colIdx] = ihbasis[:, colIdx] / np.sum(ihbasis[:, colIdx])
    # orthobas, _ = scipy.linalg.qr(ihbasis)
    ihbDF = pd.DataFrame(ihbasis, index=iht, columns=np.round(ctrs, decimals=3))
    orthobasDF = pd.DataFrame(orthobas, index=iht)
    #
    negiht = np.sort(iht[iht > 0] * (-1))
    negBDF = pd.DataFrame(0, index=negiht, columns=ihbDF.columns)
    negOrthobasDF = pd.DataFrame(0, index=negiht, columns=orthobasDF.columns)
    return pd.concat([negBDF, ihbDF]), pd.concat([negOrthobasDF, orthobasDF])


def _poisson_pseudoR2(y, yhat):
    ynull = np.mean(y)
    yhat = yhat.reshape(y.shape)
    eps = np.spacing(1)
    L1 = np.sum((y)*np.log(eps+(yhat)) - (yhat))
    # L1_v = y*np.log(eps+yhat) - yhat
    L0 = np.sum((y)*np.log(eps+(ynull)) - (ynull))
    LS = np.sum((y)*np.log(eps+(y)) - (y))
    R2 = 1-(LS-L1)/(LS-L0)
    return R2


def poisson_pseudoR2(estimator, X, y):
    # adapted from https://github.com/KordingLab/spykesMLs
    # This is our scoring function. Implements pseudo-R2
    #
    # yhat is the prediction
    # yhat = estimator.results_.predict(X)
    yhat = estimator.predict(X)
    # y null is the mean of the training data
    return _poisson_pseudoR2(y, yhat)


def _FUDE(y, yhat1, yhat2, distr='softplus'):
    # based on Goodman et al 2019
    fude = (
        1 -
        (
            pyglmnet._logL(distr, y, yhat1) /
            pyglmnet._logL(distr, y, yhat2)))
    return fude


def FUDE(estimator1, estimator2, X, y, distr='softplus'):
    yhat1 = estimator1.predict(X)
    yhat2 = estimator2.predict(X)
    return _FUDE(y, yhat1, yhat2, distr)
     
     
class trialAwareStratifiedKFold:
    def __init__(
            self, n_splits=5, shuffle=False, random_state=None,
            stratifyFactors=None, continuousFactors=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratifyFactors = stratifyFactors
        self.continuousFactors = continuousFactors

    def split(self, X, y=None, groups=None):
        trialInfo = (
            X
            .index.to_frame()
            .reset_index(drop=True)
            .loc[:, self.stratifyFactors + self.continuousFactors])
        if (self.stratifyFactors is not None):
            for idx, (name, group) in enumerate(trialInfo.groupby(self.stratifyFactors)):
                trialInfo.loc[group.index, 'stratifyGroup'] = idx
        else:
            trialInfo.loc[:, 'stratifyGroup'] = 1
        if (self.continuousFactors is not None):
            for idx, (name, group) in enumerate(trialInfo.groupby(self.continuousFactors)):
                trialInfo.loc[group.index, 'continuousGroup'] = idx
            continuousDF = trialInfo.drop_duplicates('continuousGroup')
        else:
            trialInfo.loc[:, 'continuousGroup'] = 1
            continuousDF = trialInfo
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle, random_state=self.random_state)
        folds = []
        for tr, te in skf.split(continuousDF, continuousDF['stratifyGroup']):
            trainCG = continuousDF['continuousGroup'].iloc[tr]
            trainMask = trialInfo['continuousGroup'].isin(trainCG)
            trainIdx = trialInfo.loc[trainMask, :].index.to_list()
            testCG = continuousDF['continuousGroup'].iloc[te]
            testMask = trialInfo['continuousGroup'].isin(testCG)
            testIdx = trialInfo.loc[testMask, :].index.to_list()
            folds.append((trainIdx, testIdx))
        return folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class trainTestValidationSplitter:
    def __init__(
            self, dataDF, splitterType,
            n_splits, splitterKWArgs):
        # evaluate a preliminary test-train split,
        # to get a validation-working set split
        prelimCV = splitterType(
            n_splits=n_splits + 1,
            **splitterKWArgs).split(dataDF)
        # note that train and test are iloc indices into dataDF
        (workIdx, validationIdx) = prelimCV[0]
        self.validation = validationIdx
        self.work = workIdx
        # however, prelimCV[1:] contains workIdx indices that are part of the validation set
        # here, we will re-split prelimCV's workIdx indices
        # into a train-test split
        tempCv = splitterType(
            n_splits=n_splits, **splitterKWArgs)
        folds_temp = tempCv.split(dataDF.iloc[workIdx, :])
        # folds contains iloc indices into the submatrix dataDF.iloc[workIdx, :]
        # here, we transform them to indices into the original dataDF
        originalIndices = np.arange(dataDF.index.shape[0])
        self.folds = []
        for _train, _test in folds_temp:
            train = originalIndices[_train]
            test = originalIndices[_test]
            self.folds.append((train.tolist(), test.tolist()))
        #
        self.n_splits = n_splits
        self.splitterKWArgs = splitterKWArgs
        return

    def split(self, X, y=None, groups=None):
        return self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SMWrapper(BaseEstimator, RegressorMixin):
    """
        A universal sklearn-style wrapper for statsmodels regressors
        based on https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible/
        by David Dale
    """
    def __init__(
            self, sm_class, family=None,
            alpha=None, L1_wt=None, refit=None,
            maxiter=100, tol=1e-8, disp=False
            ):
        self.sm_class = sm_class
        self.family = family
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.refit = refit
        self.maxiter = maxiter
        self.tol = tol
        self.disp = disp
        pass
    #
    def fit(self, X, y):
        model_opts = {}
        for key in dir(self):
            if key in ['family']:
                model_opts.update({key: getattr(self, key)})
        try:
            self.model_ = self.sm_class(
                y, X, **model_opts)
        except Exception:
            traceback.print_exc()
        #
        regular_opts = {}
        for key in dir(self):
            if key in ['alpha', 'L1_wt', 'refit']:
                if getattr(self, key) is not None:
                    regular_opts.update({key: getattr(self, key)})
        fit_opts = {}
        for key in dir(self):
            if key in ['maxiter', 'tol', 'disp']:
                if getattr(self, key) is not None:
                    fit_opts.update({key: getattr(self, key)})
        if not len(regular_opts.keys()):
            self.results_ = self.model_.fit(**fit_opts)
        else:
            if 'tol' in fit_opts:
                tol = fit_opts.pop('tol')
                fit_opts['cnvrg_tol'] = tol
            if 'disp' in fit_opts:
                fit_opts.pop('disp')
            self.results_ = self.model_.fit_regularized(
                **regular_opts, **fit_opts)
    #
    def predict(self, X):
        return self.results_.predict(X)
    #
    def score(self, X, y=None):
        if 'family' in dir(self):
            if 'Poisson' in str(self.family):
                return poisson_pseudoR2(self, X, y)


class pyglmnetWrapper(pyglmnet.GLM):
    def __init__(
            self, distr='poisson', alpha=0.5,
            Tau=None, group=None,
            reg_lambda=0.1,
            solver='batch-gradient',
            learning_rate=2e-1, max_iter=1000,
            tol=1e-6, eta=2.0, score_metric='deviance',
            fit_intercept=True,
            random_state=0, callback=None, verbose=False,
            track_convergence=False):
        self.track_convergence = track_convergence
        super().__init__(
            distr=distr, alpha=alpha,
            Tau=Tau, group=group,
            reg_lambda=reg_lambda,
            solver=solver,
            learning_rate=learning_rate, max_iter=max_iter,
            tol=tol, eta=eta, score_metric=score_metric,
            fit_intercept=fit_intercept,
            random_state=random_state, callback=callback, verbose=verbose)

    def _set_cv(cv, estimator=None, X=None, y=None):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        if hasattr(y, 'to_numpy'):
            yFit = y.to_numpy()
        else:
            yFit = y
        super()._set_cv(cv, estimator=estimator, X=xFit, y=yFit)

    def __repr__(self):
        return super().__repr__()

    def copy(self):
        return super().copy()

    def _prox(self, beta, thresh):
        return super()._prox(beta, thresh)

    def _cdfast(self, X, y, ActiveSet, beta, rl, fit_intercept=True):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        if hasattr(y, 'to_numpy'):
            yFit = y.to_numpy()
        else:
            yFit = y
        return super()._cdfast(
            xFit, yFit, ActiveSet, beta, rl,
            fit_intercept=fit_intercept)

    def fit(self, X, y):
        if self.track_convergence:
            dummyReg = self.copy()
            self.iterScore = []
            self.iterBetaNorm = []

            def printLoss(beta):
                if hasattr(printLoss, 'counter'):
                    printLoss.counter += 1
                else:
                    printLoss.counter = 1
                #
                dummyReg.beta_ = beta
                dummyReg.beta0_ = 0
                thisIterScore = poisson_pseudoR2(
                    dummyReg, X, y)
                self.iterScore.append(thisIterScore)
                thisBetaNorm = np.linalg.norm(beta)
                self.iterBetaNorm.append(thisBetaNorm)
                #
                if hasattr(printLoss, 'prev_beta'):
                    deltaBeta = np.linalg.norm(beta - printLoss.prev_beta)
                else:
                    deltaBeta = 0
                printLoss.prev_beta = beta
                if self.verbose:
                    print(
                        'iter {}; pR2 = {:.6f}; betaNorm = {:.6f}; deltaBeta = {:.6f}'
                        .format(printLoss.counter, thisIterScore, thisBetaNorm, deltaBeta),
                        end='\r')
            if self.callback is not None:
                self.originalCallback = self.callback
                raise(NotImplementedError)
            else:
                self.callback = printLoss
        #
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        if hasattr(y, 'to_numpy'):
            yFit = y.to_numpy()
        else:
            yFit = y
        super().fit(xFit, yFit)
        #
        if self.track_convergence:
            self.callback = None
            del dummyReg
        return self

    def predict(self, X):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        return super().predict(xFit)

    def predict_proba(self, X):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        return super().predict_proba(xFit)

    def fit_predict(self, X, y):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        if hasattr(y, 'to_numpy'):
            yFit = y.to_numpy()
        else:
            yFit = y
        return super().fit_predict(xFit, yFit)

    def score(self, X, y):
        if hasattr(X, 'to_numpy'):
            xFit = X.to_numpy()
        else:
            xFit = X
        if hasattr(y, 'to_numpy'):
            yFit = y.to_numpy()
        else:
            yFit = y
        return super().score(xFit, yFit)


class SingleNeuronRegression():
    def __init__(
            self,
            xTrain=None, yTrain=None,
            xTest=None, yTest=None,
            cv_folds=None,
            model=None, modelKWargs={},
            sampleSizeLimit=None,
            tTestAlpha=0.01,
            plotting=False, verbose=False):
        #
        self.model = model
        self.modelKWargs = modelKWargs
        #
        self.sampleSizeLimit = sampleSizeLimit
        #
        self.tTestAlpha = tTestAlpha
        #
        self.plotting = plotting
        self.verbose = verbose
        #
        if sampleSizeLimit is not None:
            for fIdx, folds in enumerate(cv_folds):
                if len(folds[0]) > sampleSizeLimit:
                    newFold = np.random.choice(
                        folds[0], size=sampleSizeLimit).tolist()
                    cv_folds[fIdx] = (newFold, folds[1])
        #
        self.cv_folds = cv_folds
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest
        #
        self.betas = pd.DataFrame(
            0,
            index=self.yTrain.columns,
            columns=self.xTrain.columns)
        #
        self.significantBetas = None
        #
        self.pvals = pd.DataFrame(
            np.nan, index=self.betas.index,
            columns=self.betas.columns)
        #
        self.regressionList = {}
        for idx, colName in enumerate(self.yTrain.columns):
            reg = self.model(**self.modelKWargs)
            self.regressionList[colName] = ({'reg': reg})
        pass
    
    def dispatchParams(self, newParams):
        for idx, colName in enumerate(self.yTrain.columns):
            self.regressionList[colName]['reg'].set_params(**newParams)
        
    def cross_val_score(self):
        #  fit the regression models
        for idx, colName in enumerate(self.yTrain.columns):
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            if hasattr(reg, 'track_convergence'):
                saveConvergenceState = reg.track_convergence
                reg.track_convergence = False
            scores = cross_val_score(
                reg, self.xTrain.to_numpy(),
                y.to_numpy(),
                error_score='raise', cv=iter(self.cv_folds))
            if hasattr(reg, 'track_convergence'):
                reg.track_convergence = saveConvergenceState
            if self.verbose:
                print('{}: mean score {}, std {}'.format(
                    colName, np.mean(scores), np.std(scores)
                    ))
            self.regressionList[colName].update({
                'cross_val_mean_test_score': np.mean(scores),
                'cross_val_std_test_score': np.std(scores),
                })
        return

    def cross_validate(self):
        #  fit the regression models
        for idx, colName in enumerate(self.yTrain.columns):
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            if hasattr(reg, 'track_convergence'):
                saveConvergenceState = reg.track_convergence
                reg.track_convergence = False
            scores = cross_validate(
                reg, self.xTrain.to_numpy(),
                y.to_numpy(), return_estimator=True,
                error_score='raise', cv=iter(self.cv_folds))
            if hasattr(reg, 'track_convergence'):
                reg.track_convergence = saveConvergenceState
            if self.verbose:
                print('{}: mean score {}, std {}'.format(
                    colName, np.mean(scores['test_score']),
                    np.std(scores['test_score'])
                    ))
            self.regressionList[colName].update({
                'cross_val_mean_test_score': np.mean(scores['test_score']),
                'cross_val_std_test_score': np.std(scores['test_score']),
                'cross_val': scores
                })
        return

    def GridSearchCV(self, gridParams):
        #  fit the regression models
        for idx, colName in enumerate(self.yTrain.columns):
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            if hasattr(reg, 'track_convergence'):
                saveConvergenceState = reg.track_convergence
                reg.track_convergence = False
            gs = GridSearchCV(
                reg, gridParams, refit=False,
                error_score='raise',
                cv=iter(self.cv_folds))
            gs.fit(self.xTrain.to_numpy(), y.to_numpy())
            if hasattr(reg, 'track_convergence'):
                reg.track_convergence = saveConvergenceState
            bestIndex = gs.best_index_
            reg.set_params(**gs.best_params_)
            self.regressionList[colName].update({
                'gridsearch_best_mean_test_score': gs.cv_results_['mean_test_score'][bestIndex],
                'gridsearch_best_std_test_score': gs.cv_results_['std_test_score'][bestIndex],
                'gridsearch': gs,
                })
            if self.verbose:
                print("Best parameters set found:\n{}".format(
                    gs.best_params_))
        return

    def fit(self):
        #  fit the regression models
        for idx, colName in enumerate(self.yTrain.columns):
            y = self.yTrain.loc[:, colName]
            reg = self.regressionList[colName]['reg']
            print('fitting model {}'.format(colName))
            seekIdx = slice(None)
            if self.sampleSizeLimit is not None:
                if self.xTrain.shape[0] > self.sampleSizeLimit:
                    seekIdx = np.random.choice(
                        self.xTrain.shape[0],
                        size=self.sampleSizeLimit).tolist()
            reg.fit(self.xTrain.iloc[seekIdx, :], y.iloc[seekIdx])
            pr2 = poisson_pseudoR2(
                reg,
                self.xTest.to_numpy(),
                self.yTest.loc[:, colName].to_numpy())
            self.regressionList[colName].update({
                'validationScore': pr2
                })
            if hasattr(reg, 'track_convergence'):
                if reg.track_convergence:
                    self.regressionList[colName]['iterScore'] = np.asarray(
                        reg.iterScore)
                    self.regressionList[colName]['iterBetaNorm'] = np.asarray(
                        reg.iterBetaNorm)
            if self.verbose:
                if hasattr(reg, 'results_'):
                    if hasattr(reg.results_, 'summary'):
                        try:
                            print(reg.results_.summary())
                            print('params \n')
                            print(reg.results_.params)
                        except Exception:
                            pass
                print('test pR2 = {}'.format(pr2))
                train_pr2 = poisson_pseudoR2(
                    reg,
                    self.xTrain.to_numpy(),
                    self.yTrain.loc[:, colName].to_numpy())
                print('train_pr2 = {}'.format(train_pr2))
            if hasattr(reg, 'results_'):
                self.betas.loc[colName, :] = reg.results_.params
                if hasattr(reg.results_, 'pvals'):
                    self.pvals.loc[colName, :] = reg.results_.pvalues
            elif hasattr(reg, 'beta_'):
                self.betas.loc[colName, :] = reg.beta_
        if hasattr(reg, 'results_'):
            if hasattr(reg.results_, 'pvalues'):
                origShape = self.pvals.shape
                flatPvals = self.pvals.to_numpy().reshape(-1)
                try:
                    _, fixedPvals, _, _ = mt(flatPvals, method='holm')
                except Exception:
                    fixedPvals = flatPvals * flatPvals.size
                self.pvals.iloc[:, :] = fixedPvals.reshape(origShape)
                self.significantBetas = self.pvals < self.tTestAlpha
        if self.significantBetas is None:
            # L1 weights encourage the parameter to go to zero;
            # assume significant by default
            self.significantBetas = self.betas.abs() > 0
        return self

    def clear_data(self):
        for idx, colName in enumerate(self.yTrain.columns):
            reg = self.regressionList[colName]['reg']
            if hasattr(reg, 'model_'):
                del reg.model_.endog, reg.model_.exog
            if hasattr(reg, 'results_'):
                if hasattr(reg.results_, 'remove_data'):
                    reg.results_.remove_data()
                if hasattr(reg.results_, 'fittedvalues'):
                    del reg.results_.fittedvalues
        del self.xTrain, self.yTrain, self.xTest, self.yTest
        return

    def plot_xy(
            self,
            showNow=False, smoothY=10, binInterval=1e-3, decimated=1,
            selT=slice(None), maxPR2=None, unitName=None, winSize=1,
            useInvLink=False, showLegend=False):
        scores = [
            {'unit': k, 'score': v['validationScore']}
            for k, v in self.regressionList.items()]
        scoresDF = pd.DataFrame(scores)
        if unitName is None:
            if maxPR2 is not None:
                uIdx = (
                    scoresDF
                    .loc[scoresDF['score'] < maxPR2, 'score']
                    .idxmax()
                    )
            else:
                uIdx = scoresDF['score'].idxmax()
            unitName = scoresDF.loc[uIdx, 'unit']
        else:
            uIdx = scoresDF.loc[scoresDF['unit'] == unitName, :].index[0]
        thisReg = self.regressionList[unitName]
        prediction = thisReg['reg'].predict(self.xTest)
        if hasattr(prediction, 'to_numpy'):
            prediction = prediction.to_numpy()
        yPlot = (self.yTest[unitName].rolling(smoothY, center=True).mean())
        fig, ax = plt.subplots(3, 1, sharex=True)
        # ax[0].plot(self.yTest[unitName].to_numpy(), label='original')
        binSize = binInterval * decimated
        countPerBin = binInterval * winSize
        xRange = np.arange(yPlot[selT].shape[0]) * binSize
        ax[0].plot(xRange, yPlot[selT].to_numpy() / countPerBin, label='smoothed original')
        ax[0].plot(xRange, prediction[selT] / countPerBin, label='prediction')
        ax[0].set_title('{}: pR^2 = {:.2f}'.format(
            unitName,
            scoresDF.loc[uIdx, 'score']))
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('(spk/s)')
        if useInvLink:
            if hasattr(thisReg['reg'], 'results_'):
                transFun = thisReg['reg'].results_.model.family.link.inverse
            else:
                transFun = lambda x: np.exp(x)
        else:
            transFun = lambda x: x
        for idx, beta in enumerate(self.betas.loc[unitName, :]):
            xPartial = beta * self.xTest.iloc[:, idx].to_numpy()
            if idx == 0:
                ax[0].plot(
                    xRange, transFun(xPartial[selT]) / countPerBin, 'r--',
                    label='{}'.format(self.xTest.columns[idx]))
            else:
                if self.significantBetas.loc[unitName, :].iloc[idx]:
                    ax[1].plot(
                        xRange, transFun(xPartial[selT]),
                        label='{}'.format(self.xTest.columns[idx]))
                else:
                    ax[2].plot(
                        xRange, xPartial[selT], ls='--',
                        label='{}'.format(self.xTest.columns[idx]))
        ax[1].set_title('p < {} regressors'.format(self.tTestAlpha))
        ax[2].set_title('p > {} regressors'.format(self.tTestAlpha))
        ax[2].set_xlabel('Time (sec)')
        if showLegend:
            for thisAx in ax:
                thisAx.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if showNow:
            plt.show()
        return fig, ax