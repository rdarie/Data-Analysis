from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
# from sklearn.decomposition import PCA
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    GridSearchCV, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection._split import _BaseKFold
# from sklearn.metrics import make_scorer
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance, MinCovDet
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.svm import LinearSVR
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pingouin as pg
from patsy.state import stateful_transform
from statsmodels.stats.multitest import multipletests as mt
import pdb, traceback
import os
from tqdm import tqdm
import scipy.optimize
from itertools import product
import joblib as jb
from joblib import Parallel, parallel_backend, delayed
from copy import copy, deepcopy
import statsmodels
import warnings
from patsy import (
    ModelDesc, EvalEnvironment, Term, Sum, Treatment, INTERCEPT,
    EvalFactor, LookupFactor, demo_data, dmatrix, dmatrices, bs)
import pyglmnet
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
import scipy.stats
from sklearn.utils.validation import check_is_fitted
from dask.distributed import Client, LocalCluster
# from sklearn.utils import _joblib, parallel_backend
from sklearn.metrics import r2_score
from ttictoc import tic, toc
import contextlib
import gc
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")

eps = np.spacing(1)
#####
## WIP durbin watson test
'''
def durbinWatson(X, E, timeVarName='bin'):
    # function [dw,pval] = durbinwatson(X,E)
    # X is the data series
    # E are the residuals
    # [n,m] = size(X);
    # n,nvars    number of variables
    # m,nobs     number of observations (time steps)
    # N,ntrials  number of trials (realisations) of a process
    #
    # % calculate Durbin Watson (DW) statistic: rule of thumb: if < 1 or > 3 then
    # % high chance of residual correlation
    #
    # dw = sum(diff(E).^2)/sum(E.^2);
    minBin = E.index.get_level_values(timeVarName).min()
    shiftedRes = E.shift(1).drop(minBin, level='bin')
    residT = E.drop(minBin, level='bin')
    residTDiff = residT - shiftedRes
    dw = (residTDiff ** 2).sum() / (X ** 2).sum()
    
def durbinWatsonCriticalValues(X, timeVarName='bin'):
    # % calculate critical values for the DW statistic using approx method (ref. [1])
    # [1] J. Durbin and G. S. Watson, "Testing for Serial Correlation in Least
    # Squares Regression I", _Biometrika_, 37, 1950.
    #
    # A = X*X';
    # covariance matrix of the data, n x n
    # X' is m x n, time on the rows, feature on the columns
    # B = filter([-1,2,-1],1,X');
    # B is a filtered version of the data
    # B([1,m],:) = (X(:,[1,m])-X(:,[2,m-1]))';
    # B is m x n
    # correct the last timestep
    # D = B/A;
    # B times A inverse, D is m x n
    # C = X*D;
    # C is m x n
    # nu1 = 2*(m-1)-trace(C);
    # nu2 = 2*(3*m-4)-2*trace(B'*D)+trace(C*C);
    # mu = nu1/(m-n);
    # sigma = sqrt(2/((m-n)*(m-n+2))*(nu2-nu1*mu));
    #
    # % evaluate p-value
    #
    # pval = normcdf(dw,mu,sigma);
    # pval = 2*min(pval,1-pval); % two tailed test
    return dw, pval
'''
def optimalSVDThreshold(Y):
    beta = Y.shape[0] / Y.shape[1]
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return omega

def ERAMittnik(
        N, maxNDim=None, endogNames=None, exogNames=None, plotting=False):
    tBins = np.unique(N.columns.get_level_values('bin'))
    nRegressors = len(endogNames + exogNames)
    H0 = pd.concat([N.shift(-nRegressors * it, axis='columns').fillna(0) for it in range(tBins.size)])
    H1 = H0.shift(-nRegressors, axis='columns').fillna(0)
    u, s, vh = np.linalg.svd(H0, full_matrices=False)
    if maxNDim is None:
        optThresh = optimalSVDThreshold(H0) * np.median(s)
        maxNDim = (s > optThresh).sum()
    stateSpaceNDim = min(maxNDim, u.shape[0])
    u = u[:, :stateSpaceNDim]
    s = s[:stateSpaceNDim]
    vh = vh[:stateSpaceNDim, :]
    O = u @ np.diag(np.sqrt(s))
    R = np.diag(np.sqrt(s)) @ vh
    F = np.diag(np.sqrt(s) ** -1) @ u.T @ H1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    C = pd.DataFrame(O[:len(endogNames), :], index=endogNames)
    C.index.name = 'target'
    C.columns.name = 'state'
    K = pd.DataFrame(R[:, :len(endogNames)], columns=endogNames)
    G = pd.DataFrame(R[:, len(endogNames):len(endogNames) + len(exogNames)], columns=exogNames)
    A = pd.DataFrame(F + K @ C)
    A.index.name = 'state'
    A.columns.name = 'state'
    D = pd.DataFrame(N.xs(0, level='bin', axis='columns').loc[:, exogNames], index=endogNames, columns=exogNames)
    D.index.name = 'target'
    D.columns.name = 'feature'
    B = pd.DataFrame(G + K @ D, columns=exogNames)
    B.index.name = 'state'
    B.columns.name = 'feature'
    if not plotting:
        return A, B, C, D
    else:
        fig, ax = plt.subplots()
        ax.plot(s)
        ax.set_ylabel('eigenvalue of H0')
        ax.axvline(maxNDim)
        return A, B, C, D, fig, ax


def getComp(uPiece, lagSamples, psiMat):
    co = uPiece.shift(lagSamples).fillna(0.) @ psiMat
    return co

def inputDrivenDynamics(
        psi=None, u=None, lag=None,
        groupVar='trialUID', joblibBackendArgs={},
        runInParallel=True, verbose=False):
    if runInParallel:
        parallelContext = jb.parallel_backend(**joblibBackendArgs)
    else:
        parallelContext = contextlib.nullcontext()
    with parallelContext:
        lOfPieces = jb.Parallel(verbose=verbose)(
            jb.delayed(getComp)(uThisTrial, lag, psi)
            for tIdx, uThisTrial in u.groupby(groupVar, sort=False))
        resDF = pd.concat(lOfPieces)
    assert (resDF.index == u.index).all()
    assert (resDF.columns == psi.columns).all()
    # resDF = resDF.loc[u.index, :]
    # resDF.columns = psi.columns
    return resDF

def ERA(
        Phi, maxNDim=None, endogNames=None, exogNames=None,
        eraReductionMethod='topFraction', eraFraction=None,
        plotting=False, nLags=None, method='ERA',
        verbose=0, noEndogTerms=False, checkApproximations=False):
    tBins = np.unique(Phi.columns.get_level_values('bin'))
    bins2Index = {bN: bIdx for bIdx, bN in enumerate(tBins)}
    Phi.rename(columns=bins2Index, inplace=True)
    # following notation from vicario 2014 (thesis)
    p = Phi.columns.get_level_values('bin').max()  # order of the varx model
    Phi.loc[:, Phi.columns.get_level_values('bin') == p] = 0
    q = len(endogNames)
    m = len(exogNames)
    if noEndogTerms:
        N = int(2 * (p + 1))
    else:
        N = nLags
    if verbose > 1:
        print('ERA(): extracting phi matrix from coefficients')
    Phi1 = {}
    Phi2 = {}
    Phi1[0] = Phi.xs(0, level='bin', axis='columns').loc[:, exogNames] # D
    D = Phi1[0]
    for j in range(1, N+1):
        if j <= p:
            Phi1[j] = Phi.xs(j, level='bin', axis='columns').loc[:, exogNames]
            Phi2[j] = Phi.xs(j, level='bin', axis='columns').loc[:, endogNames]
        else:
            Phi1[j] = Phi.xs(p, level='bin', axis='columns').loc[:, exogNames] * 0
            Phi2[j] = Phi.xs(p, level='bin', axis='columns').loc[:, endogNames] * 0
    # matrix form from Juang 1992 (p. 324)
    # lPhiMat * Psi_sDF = rPhiMat_s
    # lPhiMat * Psi_gDF = rPhiMat_g
    # # # # N = 5
    # # # # ###############
    # # # # lPhiMatLastRowList = [
    # # # #     Phi2[k].reset_index(drop=True) * (-1) for k in range(N-1, 0, -1)
    # # # # ] + [pd.DataFrame(np.eye(q))]
    # # # # lPhiMatLastRow = pd.concat(lPhiMatLastRowList, axis='columns', ignore_index=True)
    # # # # # sns.heatmap(lPhiMatLastRow.to_numpy()); plt.show()
    # # # # if verbose > 1:
    # # # #     print('ERA(): concatenating lPhiMat')
    # # # # lPhiMat = pd.concat([
    # # # #     lPhiMatLastRow.shift(k * q * (-1), axis='columns').fillna(0)
    # # # #     for k in range(N-1, -1, -1)
    # # # #     ])
    # # # # sns.heatmap(lPhiMat.to_numpy()); plt.show()
    # # # # sns.heatmap(lPhiMatLastRow.shift(250 * q * (-1), axis='columns').to_numpy()); plt.show()
    if verbose > 1:
        print('ERA(): concatenating lPhiMat')
        tIterator = tqdm(range(N-1, -1, -1), mininterval=30., maxinterval=120.)
    else:
        tIterator = range(N-1, -1, -1)
    lPhiMatRowList = []
    for kr in tIterator:
        thisRowList = (
            [Phi2[kc].reset_index(drop=True) * (-1) for kc in range(N-1-kr, 0, -1)] +
            [pd.DataFrame(np.eye(q))] +
            [pd.DataFrame(np.zeros((q, q)))] * kr
            )
        thisRow = pd.concat(thisRowList, axis='columns', ignore_index=True)
        lPhiMatRowList.append(thisRow)
        # sns.heatmap(thisRow.to_numpy()); plt.show()
    lPhiMat = pd.concat(lPhiMatRowList)
    del lPhiMatRowList
    gc.collect()
    # fig, ax = plt.subplots(1, 2); sns.heatmap(lPhiMat.to_numpy(), ax=ax[0]); sns.heatmap(lPhiMat2.to_numpy(), ax=ax[1]); plt.show()
    ###################
    if verbose > 1:
        print('ERA(): inverting lPhiMat')
    lPhiMatInv = np.linalg.inv(lPhiMat)
    if checkApproximations:
        try:
            reconstructionError = np.dot(lPhiMatInv, lPhiMat)
            assert np.allclose(reconstructionError, np.eye(*lPhiMat.shape))
        except:
            traceback.print_exc()
            print("np.max(np.abs(reconstructionError - np.eye(*lPhiMat.shape))) = {:.3e}".format(np.max(np.abs(reconstructionError - np.eye(*lPhiMat.shape)))))
    del lPhiMat
    gc.collect()
    #
    if D.any().any():
        rPhiMat_s = pd.concat({
            k: Phi1[k] + Phi2[k].to_numpy() @ D.to_numpy()
            for k in range(1, N+1)
            }, names=['bin'])
    else:
        rPhiMat_s = pd.concat({
            k: Phi1[k]
            for k in range(1, N+1)
            }, names=['bin'])
    Psi_sDF = lPhiMatInv @ rPhiMat_s
    Psi_sDF.index = rPhiMat_s.index
    Psi_sDF = Psi_sDF.unstack(level='bin')
    if verbose > 1:
        print('ERA(): sorting Psi_sDF index')
    Psi_sDF.sort_index(
        axis='columns', level='bin', sort_remaining=False,
        kind='mergesort', inplace=True)
    #
    if verbose > 1:
        print('ERA(): concatenating rPhiMat_g')
    rPhiMat_g = pd.concat({
        k: Phi2[k]
        for k in range(1, N+1)
        }, names=['bin'])
    Psi_gDF = lPhiMatInv @ rPhiMat_g
    #
    del lPhiMatInv
    gc.collect()
    #
    Psi_gDF.index = rPhiMat_g.index
    Psi_gDF = Psi_gDF.unstack(level='bin')
    if verbose > 1:
        print('ERA(): Psi_gDF.sort_index(')
    Psi_gDF.sort_index(
        axis='columns', level='bin',
        sort_remaining=False, kind='mergesort',
        inplace=True)
    if plotting:
        dt = scipy.stats.mode(np.diff(tBins))[0][0]
        fig, ax = plt.subplots(
            3, 1,
            figsize=(16, 24))
        plotS = Psi_sDF.stack('feature')
        plotG = Psi_gDF.stack('feature')
        for rowName, row in plotS.iterrows():
            ax[0].plot(row.index * dt, row, label=rowName)
        for rowName, row in plotG.iterrows():
            ax[1].plot(row.index * dt, row, label=rowName)
        ax[0].set_title('System Markov parameters (Psi_s)')
        ax[0].axvline(p * dt)
        # ax[0].legend()
        ax[1].set_title('Gain Markov parameters (Psi_g)')
        ax[1].axvline(p * dt)
        # ax[1].legend()
        ax[1].set_xlabel('Time (sec)')
    else:
        fig, ax = None, None
    if verbose > 1:
        print('ERA(): PsiDF = pd.concat([')
    PsiDF = pd.concat([Psi_gDF, Psi_sDF], axis='columns')
    del Psi_gDF, Psi_sDF
    gc.collect()
    PsiDF.sort_index(
        axis='columns', level=['bin'],
        kind='mergesort', sort_remaining=False, inplace=True)
    # PsiDF.loc[:, PsiDF.columns.get_level_values('bin') == N] = 0.
    ####
    HComps = []
    PsiBins = PsiDF.columns.get_level_values('bin')
    for hIdx in range(1, int(N / 2) + 1):
        HMask = (PsiBins >= hIdx) & (PsiBins < (hIdx + int(N / 2) - 1))
        HComps.append(PsiDF.loc[:, HMask].to_numpy())
    H0 = np.concatenate(HComps[:-1])
    H1 = np.concatenate(HComps[1:])
    if method == 'ERA':
        svdTargetH = H0
    elif method == 'ERADC':
        HH0 = H0 @ H0.T
        HH1 = H1 @ H0.T
        svdTargetH = HH0
    if verbose > 1:
        print('ERA(): svd of H')
    u, s, vh = np.linalg.svd(svdTargetH, full_matrices=False)
    if maxNDim is None:
        nonTrivialS = s[s > np.spacing(1e4)]
        if eraReductionMethod == 'optimalSVD':
            # s are sqrt eigenvalues
            optThresh = optimalSVDThreshold(svdTargetH) * np.median(nonTrivialS) #  np.median(s[:int(N)])
            maxNDim = (nonTrivialS > optThresh).sum()
        elif eraReductionMethod == 'allNonTrivial':
            maxNDim = nonTrivialS.shape[0]
        else:
            if eraFraction is None:
                eraFraction = 1 - 1e-2
            maxNDim = (np.cumsum(nonTrivialS / np.sum(nonTrivialS)) < eraFraction).sum()
    stateSpaceNDim = min(maxNDim, u.shape[0])
    fullHEigenVals = pd.Series(s)
    fullHEigenVals.index.name = 'state'
    if plotting:
        ax[2].plot(fullHEigenVals.to_numpy(), label='singular values of H')
        ax[2].set_title('singular values of Hankel matrix (ERA)')
        ax[2].set_ylabel('a. u.')
        ax[2].axvline(
            stateSpaceNDim, color='r',
            label='optimal state space n. dim. = {}'.format(stateSpaceNDim))
        ax[2].axvline(p, color='g', label='AR(p) order = {}'.format(p))
        ax[2].legend()
        ax[2].set_xlim([0, 3 * max(p, stateSpaceNDim)])
        ax[2].set_xlabel('singular value count')
    #
    u = u[:, :stateSpaceNDim].copy()
    s = s[:stateSpaceNDim].copy()
    vh = vh[:stateSpaceNDim, :].copy()
    if checkApproximations:
        hRecon = np.dot(u * s, vh)
        errorNorm = np.linalg.norm(svdTargetH - hRecon, ord='fro')
        print('ERA(): svd reconstruction error norm is {:.3g}'.format(errorNorm))
        # mse = ((svdTargetH - hRecon) ** 2).mean().mean()
    ###
    O = u @ np.diag(np.sqrt(s))
    # (u @ np.diag(np.sqrt(s))) == (u * np.sqrt(s))
    RColMask = PsiDF.columns.get_level_values('bin').isin(range(len(HComps)))
    if method == 'ERA':
        R = pd.DataFrame(np.diag(np.sqrt(s)) @ vh, index=None, columns=PsiDF.loc[:, RColMask].columns)
        ARaw = np.diag(np.sqrt(s) ** -1) @ u.T @ H1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    elif method == 'ERADC':
        R = pd.DataFrame(np.diag(np.sqrt(s)) @ u.T @ H0, index=None, columns=PsiDF.loc[:, RColMask].columns)
        ARaw = np.diag(np.sqrt(s) ** -1) @ u.T @ HH1 @ vh.T @ np.diag(np.sqrt(s) ** -1)
    ###
    A = pd.DataFrame(ARaw)
    A.index.name = 'state'
    A.columns.name = 'state'
    C = pd.DataFrame(O[:len(endogNames), :], index=endogNames)
    C.index.name = 'target'
    C.columns.name = 'state'
    BK = R.xs(1, level='bin', axis='columns')
    B = BK.loc[:, exogNames]
    B.index.name = 'state'
    B.columns.name = 'feature'
    K = BK.loc[:, endogNames]
    K.index.name = 'state'
    K.columns.name = 'target'
    ###
    outputH = pd.DataFrame(svdTargetH)
    outputH.columns.name = 'state'
    outputH.index.name = 'state'
    #
    D.index.name = 'target'
    D.columns.name = 'feature'
    #
    reconFunc = lambda nn: C @ np.linalg.matrix_power((A - K @ C), nn) @ BK
    reconPsiDF = pd.concat(
        {
            ord: reconFunc(ord - 1)
            for ord in range(1, stateSpaceNDim+1)}
        , axis='columns', names=['bin', 'feature'])
    reconPsiDF = reconPsiDF.swaplevel(axis='columns')
    if D.any().any():
        dWithBins = D.copy()
        dwbCols = dWithBins.columns.to_frame().reset_index(drop=True)
        dwbCols.loc[:, 'bin'] = 0
        dWithBins.columns = pd.MultiIndex.from_frame(dwbCols)
        fullPsiDF = pd.concat([dWithBins, PsiDF], axis='columns')
        reconPsiDF = pd.concat([dWithBins, reconPsiDF], axis='columns')
    else:
        fullPsiDF = PsiDF
    fullPsiDF.columns.names = ['feature', 'sampleBin']
    reconPsiDF.columns.names = ['feature', 'sampleBin']
    return A, B, K, C, D, outputH, fullPsiDF, reconPsiDF, fullHEigenVals, (fig, ax,)


def makeImpulseLike(
        df, categoricalCols=[], categoricalIndex=[]):
    for _, oneTrialDF in df.groupby('trialUID'):
        break
    impulseList = []
    fillIndexCols = [cN for cN in oneTrialDF.index.names if cN not in ['bin', 'trialUID', 'conditionUID']]
    fillCols = [cN for cN in oneTrialDF.columns if cN not in categoricalCols]
    if not len(categoricalCols):
        grouper = [('all', df),]
    else:
        grouper = df.groupby(categoricalCols)
    uid = 0
    for elecName, _ in grouper:
        thisImpulse = oneTrialDF.copy()
        thisTI = thisImpulse.index.to_frame().reset_index(drop=True)
        thisTI.loc[:, fillIndexCols] = 'NA'
        thisTI.loc[:, 'trialUID'] = uid
        uid += 1
        if len(categoricalCols):
            if len(categoricalCols) == 1:
                thisImpulse.loc[:, categoricalCols[0]] = elecName
                thisTI.loc[:, categoricalIndex[0]] = elecName
            else:
                for idx, cN in enumerate(categoricalCols):
                    thisImpulse.loc[:, cN] = elecName[idx]
                    thisTI.loc[:, categoricalIndex[idx]] = elecName[idx]
        #
        tBins = thisImpulse.index.get_level_values('bin')
        zeroBin = np.min(np.abs(tBins))
        thisImpulse.loc[:, fillCols] = 0.
        thisImpulse.loc[tBins == zeroBin, fillCols] = 1.
        thisImpulse.index = pd.MultiIndex.from_frame(thisTI)
        impulseList.append(thisImpulse)
    impulseDF = pd.concat(impulseList)
    return impulseDF


def getR2(srs):
    lls = float(srs.xs('llSat', level='llType'))
    lln = float(srs.xs('llNull', level='llType'))
    llf = float(srs.xs('llFull', level='llType'))
    return 1 - (lls - llf) / (lls - lln)


def partialR2(llSrs, testDesign=None, refDesign=None, designLevel='design'):
    assert testDesign is not None
    llSat = llSrs.xs(testDesign, level=designLevel).xs('llSat', level='llType')
    # llSat2 = llSrs.xs(refDesign, level='design').xs('llSat', level='llType')
    # sanity check: should be same
    # assert (llSat - llSat2).abs().max() == 0
    if refDesign is not None:
        llRef = llSrs.xs(refDesign, level=designLevel).xs('llFull', level='llType')
    else:
        llRef = llSrs.xs(testDesign, level=designLevel).xs('llNull', level='llType')
    llTest = llSrs.xs(testDesign, level=designLevel).xs('llFull', level='llType')
    # tInfoSat = llSat.index.to_frame().reset_index(drop=True)
    # tInfoRef = llRef.index.to_frame().reset_index(drop=True)
    # lhsMaskIdx refers to design, which is different between llRef and llTest
    # so the indices don't line up between llRef and llTest
    partR2 = 1 - (llSat.to_numpy() - llTest) / (llSat.to_numpy() - llRef.to_numpy())
    return partR2

def tTestNadeauCorrection(
        group, thisY, tTestArgs={}, powerAlpha=0.05,
        test_size=None):
    if 'tail' in tTestArgs:
        tail = tTestArgs['tail']
    else:
        tail = 'two-sided'
    if 'confidence' in tTestArgs:
        confidence = tTestArgs['confidence']
    else:
        confidence = 0.95
    ci_name = 'CI%.0f%%' % (100 * confidence)
    if tail == "two-sided":
        alpha = 1 - confidence
        conf = 1 - alpha / 2  # 0.975
    else:
        conf = confidence
    statsResDF = pg.ttest(group, thisY, **tTestArgs).loc['T-test', :]
    # notation from Nadeau, 2003
    J = float(statsResDF['dof']) + 1
    n2 = test_size
    n1 = 1 - test_size
    corrFactor = np.sqrt((J ** -1) / (J ** -1 + n2 / n1))
    tcrit = scipy.stats.t.ppf(conf, statsResDF['dof'])
    se = ((np.mean(group) - np.mean(np.atleast_1d(thisY))) / statsResDF['T']) / corrFactor
    statsResDF['T'] = statsResDF['T'] * corrFactor
    statsResDF[ci_name] = np.array([statsResDF['T'] - tcrit, statsResDF['T'] + tcrit]) * se
    if not isinstance(thisY, pd.Series):
        statsResDF[ci_name] += thisY
    statsResDF['cohen-d'] = statsResDF['cohen-d'] * corrFactor
    if tail == "two-sided":
        statsResDF['p-val'] = 2 * scipy.stats.t.sf(np.abs(statsResDF['T']), df=statsResDF['dof'])
    else:
        statsResDF['p-val'] = scipy.stats.t.sf(np.abs(statsResDF['T']), df=statsResDF['dof'])
    statsResDF['power'] = pg.power_ttest(
        d=statsResDF['cohen-d'], n=J, power=None, alpha=powerAlpha,
        contrast='paired', alternative=tail)
    statsResDF.drop('BF10', inplace=True)
    return statsResDF

def correctedResampledPairedTTest(
        xSrs, y=None,
        groupBy=None, tTestArgs={}, powerAlpha=0.05,
        cvIterator=None, test_size=None):
    if (cvIterator is not None) and test_size is None:
        test_size = cvIterator.splitter.sampler.test_size
    if groupBy is None:
        groupIterator = ('all', xSrs)
    else:
        groupIterator = xSrs.groupby(groupBy)
        resultsDict = {}
    for name, group in groupIterator:
        if isinstance(y, pd.Series):
            thisY = y.loc[group.index]
        else:
            thisY = y
        statsResDF = tTestNadeauCorrection(
            group, thisY, tTestArgs=tTestArgs, powerAlpha=powerAlpha,
            test_size=test_size)
        if groupBy is None:
            return statsResDF
        else:
            resultsDict[name] = statsResDF
    return pd.concat(resultsDict, axis='columns', names=groupBy).T

class raisedCosTransformer(object):
    def __init__(self, kWArgs=None):
        if kWArgs is not None:
            self.memorize_chunk(None, **kWArgs)
        return

    def memorize_chunk(
            self, vecSrs, nb=1, dt=1.,
            historyLen=None, b=1e-3,
            timeDelay=0.,
            normalize=False, useOrtho=True,
            groupBy='trialUID', tLabel='bin',
            zflag=False, logBasis=True,
            addInputToOutput=False,
            causalShift=True, causalFill=False,
            selectColumns=None, preprocFun=None,
            joblibBackendArgs=None, convolveMethod='auto', verbose=0):
        ##
        if logBasis:
            nlin = None
            invnl = None
        else:
            nlin = lambda x: x
            invnl = lambda x: x
            b = 0
        self.verbose = verbose
        self.convolveMethod = convolveMethod
        if joblibBackendArgs is not None:
            self.joblibBackendArgs = joblibBackendArgs.copy()
            if joblibBackendArgs['backend'] == 'dask':
                daskComputeOpts = self.joblibBackendArgs.pop('daskComputeOpts')
                if daskComputeOpts['scheduler'] == 'single-threaded':
                    daskClient = Client(LocalCluster(n_workers=1))
                elif daskComputeOpts['scheduler'] == 'processes':
                    daskClient = Client(LocalCluster(processes=True))
                elif daskComputeOpts['scheduler'] == 'threads':
                    daskClient = Client(LocalCluster(processes=False))
                else:
                    print('Scheduler name is not correct!')
                    daskClient = Client()
        else:
            self.joblibBackendArgs = None
        self.nb = nb
        self.dt = dt
        self.historyLen = historyLen
        self.zflag = zflag
        self.logBasis = logBasis
        self.normalize = normalize
        self.useOrtho = useOrtho
        self.timeDelay = timeDelay
        self.groupBy = groupBy
        self.tLabel = tLabel
        self.addInputToOutput = addInputToOutput
        self.selectColumns = selectColumns
        if preprocFun is None:
            self.preprocFun = lambda x: x
        else:
            self.preprocFun = preprocFun
        self.causalShift = causalShift
        self.causalFill = causalFill
        if not logBasis:
            b = 0.
        self.b = b
        self.endpoints = raisedCosBoundary(
            b=b, DT=historyLen,
            minX=0.,
            nb=nb, nlin=nlin, invnl=invnl, causal=causalShift)
        if logBasis:
            self.ihbasisDF, self.orthobasisDF = makeLogRaisedCosBasis(
                nb=nb, dt=dt, endpoints=self.endpoints, b=b,
                zflag=zflag, normalize=normalize, causal=causalFill)
        else:
            self.ihbasisDF, self.orthobasisDF = makeRaisedCosBasis(
                nb=nb, dt=dt, endpoints=self.endpoints,
                normalize=normalize, causal=causalFill)
        self.iht = np.asarray(self.ihbasisDF.index)
        if self.timeDelay != 0:
            extraSamples = int(self.timeDelay / self.dt)
            extraTime = self.iht[-1] + self.dt * np.arange(1, extraSamples + 1)
            self.iht = np.concatenate([self.iht, extraTime])
            extraPadding = pd.DataFrame(0, columns=self.ihbasisDF.columns, index=extraTime)
            self.ihbasisDF = pd.concat([extraPadding, self.ihbasisDF])
            self.ihbasisDF.index = self.iht
            self.orthobasisDF = pd.concat([extraPadding, self.orthobasisDF])
            self.orthobasisDF.index = self.iht
        if self.useOrtho:
            self.basisDF = self.orthobasisDF
        else:
            self.basisDF = self.ihbasisDF
        self.leftShiftBasis = int(((max(self.iht) - min(self.iht)) / 2 + min(self.iht)) / self.dt) + 1

        def transformPiece(name, group):
            resDF = pd.DataFrame(np.nan, index=group.index, columns=self.basisDF.columns)
            for cNIdx, cN in enumerate(self.basisDF.columns):
                resCN = resDF.columns[cNIdx]
                sig = self.preprocFun(group)
                convResult = scipy.signal.convolve(
                    sig.to_numpy(),
                    self.basisDF[cN].to_numpy(),
                    mode='full', method=self.convolveMethod)
                leftSeek = max(
                    int(convResult.shape[0] / 2 - group.shape[0] / 2 - self.leftShiftBasis), 0)
                rightSeek = leftSeek + group.shape[0]
                convResult = convResult[leftSeek:rightSeek]
                resDF.loc[group.index, resCN] = convResult
            return resDF
        self.transformPiece = transformPiece
        return

    def memorize_finish(self):
        return

    def transform(
            self, vecSrs, nb=1, dt=1.,
            historyLen=None, b=1e-3,
            normalize=False, useOrtho=True,
            timeDelay=0.,
            groupBy='trialUID', tLabel='bin',
            zflag=False, logBasis=True, causalShift=True, causalFill=False,
            addInputToOutput=False,
            selectColumns=None, preprocFun=None,
            convolveMethod='auto', joblibBackendArgs=None, verbose=0):
        # print('Starting to apply raised cos basis to {} (size={})'.format(vecSrs.name, vecSrs.size))
        # for line in traceback.format_stack():
        #     print(line.strip())
        columnNames = ['{}_{}'.format(vecSrs.name, basisCN) for basisCN in self.basisDF.columns]
        # resDF = pd.DataFrame(np.nan, index=vecSrs.index, columns=columnNames)
        '''
            lOfPieces = []
            for name, group in vecSrs.groupby(self.groupBy):
                lOfPieces.append(self.transformPiece(name, group))
            '''
        if self.joblibBackendArgs is None:
            contextManager = contextlib.nullcontext()
        else:
            contextManager = parallel_backend(**self.joblibBackendArgs)
        if self.verbose >= 1:
            if self.verbose >= 2:
                print('Analyzing signal {}'.format(vecSrs.name))
                print('joblibBackendArgs = {}'.format(self.joblibBackendArgs))
                print('joblib context manager = {}'.format(contextManager))
            #  tIterator = tqdm(vecSrs.groupby(self.groupBy), mininterval=30., maxinterval=120.)
            tIterator = vecSrs.groupby(self.groupBy)
        else:
            tIterator = vecSrs.groupby(self.groupBy)
        with contextManager:
            lOfPieces = Parallel(verbose=self.verbose)(
                delayed(self.transformPiece)(name, group)
                for name, group in tIterator)
            resDF = pd.concat(lOfPieces)
            resDF = resDF.loc[vecSrs.index, :]
            resDF.columns = columnNames
        if self.addInputToOutput:
            sig = self.preprocFun(vecSrs)
            resDF.insert(0, 0., sig)
        if self.selectColumns is not None:
            resDF = resDF.iloc[:, self.selectColumns]
        #
        return resDF

    def plot_basis(self):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(self.ihbasisDF)
        titleStr = 'raised log cos basis' if self.logBasis else 'raised cos basis'
        ax[0].set_title(titleStr)
        ax[1].plot(self.orthobasisDF)
        ax[1].set_title('orthogonalized basis')
        ax[1].set_xlabel('Time (sec)')
        return fig, ax

patsyRaisedCosTransformer = stateful_transform(raisedCosTransformer)

class bSplineTransformer(object):
    def __init__(self, kWArgs=None):
        if kWArgs is not None:
            self.memorize_chunk(None, **kWArgs)
        return

    def memorize_chunk(
            self, vecSrs, nb=1, dt=1.,
            historyLen=None, timeDelay=0.,
            normalize=False, useOrtho=True,
            useFirst=False, useLast=False,
            groupBy='trialUID', tLabel='bin',
            addInputToOutput=False,
            causalShift=True, causalFill=False,
            selectColumns=None, preprocFun=None,
            joblibBackendArgs=None, convolveMethod='auto', verbose=0):
        ##
        self.verbose = verbose
        self.convolveMethod = convolveMethod
        if joblibBackendArgs is not None:
            self.joblibBackendArgs = joblibBackendArgs.copy()
            if joblibBackendArgs['backend'] == 'dask':
                daskComputeOpts = self.joblibBackendArgs.pop('daskComputeOpts')
                if daskComputeOpts['scheduler'] == 'single-threaded':
                    daskClient = Client(LocalCluster(n_workers=1))
                elif daskComputeOpts['scheduler'] == 'processes':
                    daskClient = Client(LocalCluster(processes=True))
                elif daskComputeOpts['scheduler'] == 'threads':
                    daskClient = Client(LocalCluster(processes=False))
                else:
                    print('Scheduler name is not correct!')
                    daskClient = Client()
        else:
            self.joblibBackendArgs = None
        self.nb = nb
        self.dt = dt
        self.normalize = normalize
        self.useOrtho = useOrtho
        self.useFirst = useFirst
        self.useLast = useLast
        self.groupBy = groupBy
        self.tLabel = tLabel
        self.addInputToOutput = addInputToOutput
        self.selectColumns = selectColumns
        if preprocFun is None:
            self.preprocFun = lambda x: x
        else:
            self.preprocFun = preprocFun
        self.causalShift = causalShift
        if causalShift:
            self.timeDelay = timeDelay + dt
        else:
            self.timeDelay = timeDelay
        self.endpoints = [self.timeDelay, self.timeDelay + historyLen]
        #
        self.causalFill = causalFill
        self.ihbasisDF, self.orthobasisDF = makeBSplineBasis(
            nb=nb, dt=dt, endpoints=self.endpoints,
            useFirst=useFirst, useLast=useLast,
            normalize=normalize, causal=causalFill)
        self.iht = np.asarray(self.ihbasisDF.index)
        if self.useOrtho:
            self.basisDF = self.orthobasisDF
        else:
            self.basisDF = self.ihbasisDF
        self.leftShiftBasis = int(((max(self.iht) - min(self.iht)) / 2 + min(self.iht)) / self.dt) + 1
        self.historyLen = self.basisDF.index[-1]

        def transformPiece(name, group):
            resDF = pd.DataFrame(np.nan, index=group.index, columns=self.basisDF.columns)
            for cNIdx, cN in enumerate(self.basisDF.columns):
                resCN = resDF.columns[cNIdx]
                sig = self.preprocFun(group)
                convResult = scipy.signal.convolve(
                    sig.to_numpy(),
                    self.basisDF[cN].to_numpy(),
                    mode='full', method=self.convolveMethod)
                leftSeek = max(
                    int(convResult.shape[0] / 2 - group.shape[0] / 2 - self.leftShiftBasis), 0)
                rightSeek = leftSeek + group.shape[0]
                convResult = convResult[leftSeek:rightSeek]
                resDF.loc[group.index, resCN] = convResult
            return resDF
        self.transformPiece = transformPiece
        return

    def memorize_finish(self):
        return

    def transform(
            self, vecSrs, nb=1, dt=1.,
            historyLen=None, normalize=False, useOrtho=True,
            timeDelay=0., groupBy='trialUID', tLabel='bin',
            causalShift=True, causalFill=False,
            useFirst=False, useLast=False,
            addInputToOutput=False, selectColumns=None, preprocFun=None,
            convolveMethod='auto', joblibBackendArgs=None, verbose=0):
        # print('Starting to apply raised cos basis to {} (size={})'.format(vecSrs.name, vecSrs.size))
        # for line in traceback.format_stack():
        #     print(line.strip())
        columnNames = ['{}_{}'.format(vecSrs.name, basisCN) for basisCN in self.basisDF.columns]
        # resDF = pd.DataFrame(np.nan, index=vecSrs.index, columns=columnNames)
        '''
            lOfPieces = []
            for name, group in vecSrs.groupby(self.groupBy):
                lOfPieces.append(self.transformPiece(name, group))
            '''
        if self.joblibBackendArgs is None:
            contextManager = contextlib.nullcontext()
        else:
            contextManager = parallel_backend(**self.joblibBackendArgs)
        if self.verbose >= 1:
            if self.verbose >= 2:
                print('Analyzing signal {}'.format(vecSrs.name))
                print('joblibBackendArgs = {}'.format(self.joblibBackendArgs))
                print('joblib context manager = {}'.format(contextManager))
            #  tIterator = tqdm(vecSrs.groupby(self.groupBy), mininterval=30., maxinterval=120.)
            tIterator = vecSrs.groupby(self.groupBy)
        else:
            tIterator = vecSrs.groupby(self.groupBy)
        with contextManager:
            lOfPieces = Parallel(verbose=self.verbose)(
                delayed(self.transformPiece)(name, group)
                for name, group in tIterator)
            resDF = pd.concat(lOfPieces)
            resDF = resDF.loc[vecSrs.index, :]
            resDF.columns = columnNames
        if self.addInputToOutput:
            sig = self.preprocFun(vecSrs)
            resDF.insert(0, 0., sig)
        if self.selectColumns is not None:
            resDF = resDF.iloc[:, self.selectColumns]
        #
        return resDF

    def plot_basis(self):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(self.ihbasisDF)
        titleStr = 'b spline basis'
        ax[0].set_title(titleStr)
        ax[1].plot(self.orthobasisDF)
        ax[1].set_title('orthogonalized basis')
        ax[1].set_xlabel('Time (sec)')
        return fig, ax

patsyBSplineTransformer = stateful_transform(bSplineTransformer)

class DataFrameAverager(TransformerMixin, BaseEstimator):
    def __init__(
            self, stimConditionNames=None,
            addIndexFor=None, burnInPeriod=None):
        self.stimConditionNames = stimConditionNames
        self.addIndexFor = addIndexFor
        self.burnInPeriod = burnInPeriod
    #
    def fit(self, X, y=None):
        return self
    #
    def transform(self, X):
        averagedX = X.groupby(self.stimConditionNames).mean()
        if self.addIndexFor is not None:
            trialInfo = averagedX.groupby(self.addIndexFor).mean().index
            lookup = pd.Series(np.arange(trialInfo.size), index=trialInfo)
            newIndexFrame = averagedX.index.to_frame()
            newEntries = newIndexFrame.set_index(self.addIndexFor).index.map(lookup)
            newIndexFrame.loc[:, 'trialUID'] = newEntries.to_numpy()
            averagedX.index = pd.MultiIndex.from_frame(newIndexFrame.reset_index(drop=True))
        if self.burnInPeriod is not None:
            tBins = averagedX.index.get_level_values('bin')
            burnInMask = tBins > (tBins.min() + self.burnInPeriod)
            averagedX = averagedX.loc[burnInMask, :]
        return averagedX

    def inverse_transform(self, X):
        return X


class DataFrameBinTrimmer(TransformerMixin, BaseEstimator):
    def __init__(
            self, burnInPeriod=None):
        self.burnInPeriod = burnInPeriod
        return
    #
    def fit(self, X, y=None):
        return self
    #
    def transform(self, X):
        if self.burnInPeriod is not None:
            tBins = X.index.get_level_values('bin')
            burnInMask = tBins > (tBins.min() + self.burnInPeriod)
            X = X.loc[burnInMask, :]
        return X
    #
    def inverse_transform(self, X):
        return X

class DataFrameColumnAverager(TransformerMixin, BaseEstimator):
    def __init__(self, groupBy=None, metaDataFiller={}):
        self.groupBy = groupBy
        self.metaDataFiller = metaDataFiller
        return
    #
    def fit(self, X, y=None):
        return self
    #
    def transform(self, X):
        if self.groupBy is None:
            self.groupBy = 'all'
        res = X.groupby(self.groupBy, axis='columns').mean()
        resColumns = res.columns.to_frame().reset_index(drop=True)
        for cN in X.columns.names:
            if cN not in res.columns.names:
                if cN in self.metaDataFiller:
                    resColumns.loc[:, cN] = self.metaDataFiller[cN]
                else:
                    resColumns.loc[:, cN] = 'NA'
        res.columns = pd.MultiIndex.from_frame(resColumns.loc[:, X.columns.names])
        return res
    #
    def inverse_transform(self, X):
        return X

class DataFramePassThrough(TransformerMixin, BaseEstimator):
    def __init__(self):
        return
    #
    def fit(self, X, y=None):
        return self
    #
    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class DataFrameMetaDataToColumns(TransformerMixin, BaseEstimator):
    def __init__(self, addColumns=[]):
        self.addColumns = addColumns
        return
    #
    def fit(self, X, y=None):
        return self
    #
    def transform(self, X):
        trialInfo = X.index.to_frame().reset_index(drop=True)
        Xout = X.copy()
        for cN in self.addColumns:
            cNKey = (cN, 0,) + ('NA',) * 4
            Xout.loc[:, cNKey] = trialInfo.loc[:, cN].to_numpy()
        return Xout

    def inverse_transform(self, X):
        maskColumns = X.columns.get_level_values('feature').isin(self.addColumns).to_numpy()
        Xout = X.copy()
        Xout = Xout.loc[:, ~maskColumns]
        return Xout


class dummyFold:
    def __init__(self, folds):
        self.folds = folds
        self.n_splits = len(folds)

    def split(self, X, y=None, groups=None):
        return self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class trialAwareStratifiedKFold:
    def __init__(
            self, samplerClass=None, samplerKWArgs=None,
            resamplerClass=None, resamplerKWArgs={},
            stratifyFactors=None, continuousFactors=None):
        if samplerClass is None:
            samplerClass = StratifiedShuffleSplit
        if len(samplerKWArgs.keys()) == 0:
            samplerKWArgs = dict(
                n_splits=7, random_state=None,
                test_size=None,)
        self.sampler = samplerClass(**samplerKWArgs)
        self.samplerKWArgs = samplerKWArgs
        self.stratifyFactors = stratifyFactors
        self.continuousFactors = continuousFactors
        if resamplerClass is not None:
            self.resampler = resamplerClass(**resamplerKWArgs)
        else:
            self.resampler = None
        self.folds = []
        self.raw_folds = []
        self.folds_per_trial = []
        self.raw_folds_per_trial = []
        self.continuousGroup = None

    def fit(self, X, y=None, groups=None):
        #
        trialMetadata = (
            X
            .index.to_frame()
            .reset_index(drop=True)
            .loc[:, self.stratifyFactors + self.continuousFactors])
        if (self.continuousFactors is not None):
            infoPerTrial = (
                trialMetadata
                .drop_duplicates(subset=self.continuousFactors)
                .copy())
        else:
            infoPerTrial = trialMetadata.copy()
        #
        infoPerTrial.reset_index(drop=True)
        infoPerTrial.loc[:, 'continuousGroup'] = np.arange(infoPerTrial.shape[0])
        #
        if (self.continuousFactors is not None):
            mappingInfo = infoPerTrial.set_index(self.continuousFactors)['continuousGroup']
            trialMetadata.loc[:, 'continuousGroup'] = (
                trialMetadata.set_index(self.continuousFactors).index.map(mappingInfo))
        else:
            trialMetadata.loc[:, 'continuousGroup'] = np.arange(trialMetadata.shape[0])
        self.continuousGroup = trialMetadata['continuousGroup'].to_numpy()
        #
        def cgLookup(x):
            return trialMetadata.index[trialMetadata['continuousGroup'] == x]
        #
        if (self.stratifyFactors is not None):
            labelsPerTrial = infoPerTrial.loc[:, self.stratifyFactors]
            stratifyGroup = pd.DataFrame(np.nan, index=infoPerTrial.index, columns=['stratifyGroup'])
            for idx, (name, group) in enumerate(labelsPerTrial.groupby(self.stratifyFactors)):
                stratifyGroup.loc[group.index, :] = idx
            ## check that there are no single element groups
            sgvc = stratifyGroup['stratifyGroup'].value_counts()
            badIndices = stratifyGroup.index[stratifyGroup['stratifyGroup'].isin(sgvc.index[sgvc < 2].to_numpy())]
            if not badIndices.empty:
                print('trialAwareStratifiedKFold: ignoring single element groups:\n{}'.format(labelsPerTrial.loc[badIndices, :]))
                infoPerTrial.drop(badIndices, inplace=True)
                stratifyGroup.drop(badIndices, inplace=True)
        else:
            labelsPerTrial = pd.DataFrame(
                np.ones((infoPerTrial.shape[0], 1)),
                index=infoPerTrial.index, columns=['stratifyGroup'])
            stratifyGroup = labelsPerTrial.copy()
        #
        requestedZeroTestSize = (self.sampler.test_size == 0)
        notEnoughSamples = np.around(stratifyGroup.shape[0] * self.sampler.test_size) < stratifyGroup['stratifyGroup'].nunique()
        # pdb.set_trace()
        if (notEnoughSamples) and (not requestedZeroTestSize):
            print('#' * 50)
            print('Setting sampler to ShuffleSplit')
            print('#' * 50)
            samplerClass = ShuffleSplit
            self.sampler = samplerClass(**self.samplerKWArgs)
        #
        self.folds = []
        self.raw_folds = []
        self.folds_per_trial = []
        self.raw_folds_per_trial = []
        if requestedZeroTestSize:
            print('#' * 50)
            print('Setting  test size to 0')
            print('#' * 50)
            sampleIterator = [
                (np.arange(infoPerTrial.shape[0], dtype=np.int), np.asarray([], dtype=np.int))
                for idx in range(self.sampler.n_splits)]
        else:
            try:
                sampleIterator = self.sampler.split(infoPerTrial, stratifyGroup)
            except Exception:
                print('#' * 50)
                traceback.print_exc()
                print('#' * 50)
                sampleIterator = [
                    (np.arange(infoPerTrial.shape[0], dtype=np.int), np.asarray([], dtype=np.int))
                    for idx in range(self.sampler.n_splits)]
        for tr, te in sampleIterator:
            trainCG = infoPerTrial['continuousGroup'].iloc[tr].reset_index(drop=True)
            rawTrainCG = trainCG.to_list()
            rawTrainIdx = np.concatenate(trainCG.apply(cgLookup).to_list())
            if self.resampler is not None:
                X_res, _ = self.resampler.fit_resample(trainCG.to_numpy().reshape(-1, 1), stratifyGroup.iloc[tr])
                trainCG = pd.Series(X_res.flatten())
            trainIdx = np.concatenate(trainCG.apply(cgLookup).to_list())
            #
            testCG = infoPerTrial['continuousGroup'].iloc[te].reset_index(drop=True)
            rawTestCG = []
            if requestedZeroTestSize or notEnoughSamples:
                rawTestIdx = np.asarray([], dtype=np.int)
                testIdx = np.asarray([], dtype=np.int)
            else:
                rawTestCG = testCG.to_list()
                rawTestIdx = np.concatenate(testCG.apply(cgLookup).to_list())
                if self.resampler is not None:
                    X_res2, _ = self.resampler.fit_resample(testCG.to_numpy().reshape(-1, 1), stratifyGroup.iloc[te])
                    testCG = pd.Series(X_res2.flatten())
                testIdx = np.concatenate(testCG.apply(cgLookup).to_list())
            self.folds.append((trainIdx, testIdx))
            self.raw_folds.append((rawTrainIdx, rawTestIdx))
            self.folds_per_trial.append((trainCG.to_list(), testCG.to_list()))
            self.raw_folds_per_trial.append((rawTrainCG, rawTestCG))

    def split(self, X, y=None, groups=None):
        return self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


class trainTestValidationSplitter:
    def __init__(
            self, dataDF=None, n_splits=5,
            samplerClass=None, samplerKWArgs=None,
            resamplerClass=None, resamplerKWArgs={},
            prelimSplitterClass=None, prelimSplitterKWArgs={},
            splitterClass=None, splitterKWArgs={}):
        if splitterClass is None:
            splitterClass = trialAwareStratifiedKFold
        if prelimSplitterClass is None:
            prelimSplitterClass = splitterClass
        #######
        if len(prelimSplitterKWArgs.keys()) == 0:
            prelimSplitterKWArgs = splitterKWArgs
        ########
        if resamplerClass is not None:
            prelimSplitterKWArgs.update(
                {'resamplerClass': resamplerClass,
                 'resamplerKWArgs': resamplerKWArgs}
            )
            splitterKWArgs.update(
                {'resamplerClass': resamplerClass,
                 'resamplerKWArgs': resamplerKWArgs}
            )
        #####
        # if samplerClass is None:
        #     samplerClass = StratifiedShuffleSplit
        # if len(samplerKWArgs.keys()) == 0:
        #     samplerKWArgs = dict(random_state=None)
        # ###
        # prelimSplitterKWArgs.update(dict(samplerClass=samplerClass, samplerKWArgs=samplerKWArgs,))
        prelimSplitterKWArgs['samplerKWArgs']['n_splits'] = n_splits + 1
        # if samplerClass == StratifiedShuffleSplit:
        #     prelimSplitterKWArgs['samplerKWArgs']['test_size'] = 1 / (n_splits + 1)
        # evaluate a preliminary test-train split,
        # to get a validation-working set split
        self.prelimSplitter = prelimSplitterClass(**prelimSplitterKWArgs)
        self.prelimSplitter.fit(dataDF)
        # note that train and test are iloc indices into dataDF
        (workIdx, validationIdx) = self.prelimSplitter.folds[0]
        # NB! workIdx, validationIdx might have repeats or missing values if we are resampling
        self.work, self.validation = self.prelimSplitter.raw_folds[0]
        self.workIterator = dummyFold(folds=[(workIdx, validationIdx,)])
        # however, prelimCV[1:] contains workIdx indices that are part of the validation set
        # here, we will re-split prelimCV's workIdx indices
        # into a train-test split
        # splitterKWArgs.update(dict(samplerClass=samplerClass, samplerKWArgs=samplerKWArgs,))
        splitterKWArgs['samplerKWArgs']['n_splits'] = n_splits
        # if samplerClass == StratifiedShuffleSplit:
        #     splitterKWArgs['samplerKWArgs']['test_size'] = 1 / n_splits
        self.splitter = splitterClass(**splitterKWArgs)
        self.splitter.fit(dataDF.iloc[self.work, :])
        # folds contains iloc indices into the submatrix dataDF.iloc[self.work, :]
        # here, we transform them to indices into the original dataDF
        originalIndices = np.arange(dataDF.shape[0])[self.work]
        self.folds = []
        self.raw_folds = []
        for foldIdx, (_train, _test) in enumerate(self.splitter.folds):
            train = originalIndices[self.splitter.folds[foldIdx][0]]
            test = originalIndices[self.splitter.folds[foldIdx][1]]
            self.folds.append((train.tolist(), test.tolist()))
            #
            raw_train = originalIndices[self.splitter.raw_folds[foldIdx][0]]
            raw_test = originalIndices[self.splitter.raw_folds[foldIdx][1]]
            self.raw_folds.append((raw_train.tolist(), raw_test.tolist()))
        #
        self.n_splits = n_splits
        self.splitterKWArgs = splitterKWArgs
        return

    def split(self, X, y=None, groups=None):
        return self.folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def plot_schema(self):
        fig, ax = plt.subplots()
        for foldIdx, (tr, te) in enumerate(self.folds):
            lhtr, = ax.plot(tr, [foldIdx for j in tr], 'bo', alpha=0.3, label='train')
            lhte, = ax.plot(te, [foldIdx for j in te], 'ro', alpha=0.3, label='test')
        maxFoldIdx = foldIdx + 1
        for foldIdx, (tr, te) in enumerate(self.workIterator.folds):
            lhw, = ax.plot(tr, [(foldIdx + maxFoldIdx) for j in tr], 'co', alpha=0.3, label='work')
            lhv, = ax.plot(te, [(foldIdx + maxFoldIdx) for j in te], 'mo', alpha=0.3, label='validation')
        ax.legend(handles=[lhtr, lhte, lhw, lhv])
        return fig, ax


def crossValidationScores(
        X, y=None,
        estimatorClass=None, estimatorInstance=None,
        estimatorKWArgs={}, crossvalKWArgs={},
        joblibBackendArgs=None, verbose=0):
    if verbose > 0:
        print('joblibBackendArgs = {}'.format(joblibBackendArgs))
    #
    if joblibBackendArgs is None:
        contextManager = contextlib.nullcontext()
    else:
        contextManager = parallel_backend(**joblibBackendArgs)
    if verbose > 0:
        print('crossValidationScores() using contextManager: {}'.format(contextManager))
    with contextManager:
        if estimatorInstance is None:
            estimatorInstance = estimatorClass(**estimatorKWArgs)
        if hasattr(crossvalKWArgs['cv'], 'workIterator'):
            workEstim = clone(estimatorInstance)
        scores = cross_validate(
            estimatorInstance, X, y, verbose=verbose, **crossvalKWArgs)
        # train on all of the "working" samples, eval on the "validation"
        if hasattr(crossvalKWArgs['cv'], 'workIterator'):
            workingIdx = crossvalKWArgs['cv'].work
            validIdx = crossvalKWArgs['cv'].validation
            workingIdx, validIdx = crossvalKWArgs['cv'].workIterator.split(X)[0]
            #
            workX = X.iloc[workingIdx, :]
            valX = X.iloc[validIdx, :]
            if y is not None:
                workY = y.iloc[workingIdx]
                valY = y.iloc[validIdx]
            else:
                workY = None
                valY = None
            #
            workEstim.fit(workX, workY)
            #
            scores['fit_time'] = np.append(scores['fit_time'], np.nan)
            scores['score_time'] = np.append(scores['score_time'], np.nan)
            if 'estimator' in scores:
                scores['estimator'].append(workEstim)
            if 'scoring' in crossvalKWArgs:
                scoreFun = crossvalKWArgs['scoring']
                scores['train_score'] = np.append(scores['train_score'], scoreFun(workEstim, workX, workY))
                scores['test_score'] = np.append(scores['test_score'], scoreFun(workEstim, valX, valY))
            else:
                scores['train_score'] = np.append(scores['train_score'], workEstim.score(workX, workY))
                scores['test_score'] = np.append(scores['test_score'], workEstim.score(valX, valY))
    return scores


def gridSearchHyperparameters(
        X, y=None,
        estimatorClass=None, estimatorInstance=None,
        estimatorKWArgs={}, gridSearchKWArgs={},
        crossvalKWArgs={}, joblibBackendArgs={},
        useElasticNetCV=False,
        verbose=0, recalculateBestEstimator=False, timeThis=False):
    #
    workGridSearchKWArgs = deepcopy(gridSearchKWArgs)
    workGridSearchKWArgs['cv'] = gridSearchKWArgs['cv'].workIterator
    if timeThis:
        tic()
    if joblibBackendArgs is None:
        contextManager = contextlib.nullcontext()
    else:
        contextManager = parallel_backend(**joblibBackendArgs)
    with contextManager:
        if estimatorInstance is None:
            estimatorInstance = estimatorClass(**estimatorKWArgs)
        if isinstance(estimatorInstance, ElasticNet) and useElasticNetCV:
            gridSearcher = ElasticNetCV(
                verbose=verbose, **gridSearchKWArgs)
            workGridSearcher = ElasticNetCV(
                verbose=verbose, **workGridSearchKWArgs)
        else:
            gridSearcher = GridSearchCV(
                estimatorInstance, verbose=verbose, **gridSearchKWArgs)
            workGridSearcher = GridSearchCV(
                estimatorInstance, verbose=verbose, **workGridSearchKWArgs)
        if verbose > 0:
            print('Fitting gridSearchCV...')
            print('X.dtypes = {}'.format(X.dtypes))
            print('y.dtypes = {}'.format(y.dtypes))
        gridSearcher.fit(X, y)
        workGridSearcher.fit(X, y)
    if timeThis:
        print('Elapsed time: {}'.format(toc()))
    if verbose:
        print('    Done fitting gridSearchCV!')
    gsScoresDF = pd.DataFrame(gridSearcher.cv_results_)
    paramColNames = []
    paramColNamesShort = []
    colRenamer = {}
    for cN in gsScoresDF.columns:
        if cN.startswith('param_'):
            paramColNames.append(cN)
            paramColNamesShort.append(cN[6:])
            colRenamer[cN] = cN[6:]
    scoresDict = {}
    for foldIdx in range(gridSearcher.n_splits_):
        splitName = 'split{}_'.format(foldIdx)
        theseColNames = paramColNames + ['{}test_score'.format(splitName)]
        colRenamer['{}test_score'.format(splitName)] = 'test_score'
        trainField = '{}train_score'.format(splitName)
        if trainField in gsScoresDF.copy():
            theseColNames += [trainField]
            colRenamer[trainField] = 'train_score'
        theseScores = gsScoresDF.loc[:, theseColNames]
        theseScores.rename(columns=colRenamer, inplace=True)
        theseScores.set_index(paramColNamesShort, inplace=True)
        scoresDict[foldIdx] = theseScores
    prelimGsCVResultsDF = pd.concat(scoresDict, names=['fold'])
    optParams = None
    if recalculateBestEstimator:
        # the estimator that maximizes the score
        # might yield similar results to other estimators
        # that are much less complicated
        #
        # Here, we find a confidence interval around the score
        # and choose a model that is "close enough" to the max
        scoreMean = prelimGsCVResultsDF.groupby(paramColNamesShort).mean()['test_score']
        maxScore, minScore = scoreMean.max(), scoreMean.min()
        scoreSem = prelimGsCVResultsDF.groupby(paramColNamesShort).sem()['test_score']
        threshold = minScore + (maxScore - minScore) * 0.95 - scoreSem.loc[scoreMean.idxmax()]
        if scoreMean.index.names == ['dim_red__n_components']:
            optParams = {'dim_red__n_components': scoreMean.loc[scoreMean > threshold].index.min()}
            print('Resetting optimal params to:\n{}\n'.format(optParams))
        # else.. depends on whether we want the params to be maximized or minimized TODO
    workGsScoresDF = pd.DataFrame(workGridSearcher.cv_results_).rename(columns=colRenamer)
    workGsScoresDF = workGsScoresDF.loc[:, paramColNamesShort + ['test_score', 'train_score']]
    workGsScoresDF.set_index(paramColNamesShort, inplace=True)
    scoresDict[gridSearcher.n_splits_] = workGsScoresDF
    gsCVResultsDF = pd.concat(scoresDict, names=['fold'])
    if optParams is None:
        if isinstance(estimatorInstance, ElasticNet) and useElasticNetCV:
            optParams = {
                'alpha': gridSearcher.alpha_,
                'l1_ratio': gridSearcher.l1_ratio_}
        else:
            optParams = gridSearcher.best_params_
    '''optKeys = []
    optVals = []
    for k, v in optParams.items():
        optKeys.append(k)
        optVals.append(v)
    scores = gsCVResultsDF.xs(optVals, level=optKeys)'''
    scoringEstimatorParams = copy(estimatorKWArgs)
    scoringEstimatorParams.update(optParams)
    if verbose:
        print('cross val scoring')
    if estimatorClass is None:
        estimatorInstanceForCrossVal = clone(estimatorInstance)
        estimatorInstanceForCrossVal.set_params(**scoringEstimatorParams)
    else:
        estimatorInstanceForCrossVal = None
    scores = crossValidationScores(
        X, y=y, estimatorInstance=estimatorInstanceForCrossVal,
        estimatorClass=estimatorClass,
        estimatorKWArgs=scoringEstimatorParams,
        crossvalKWArgs=crossvalKWArgs,
        joblibBackendArgs=joblibBackendArgs,
        verbose=verbose)
    if verbose:
        print('    Done cross val scoring!')
    return scores, gridSearcher, gsCVResultsDF


def scoreColumnTransformer(estimator, X, y=None):
    componentScores = []
    for trfTuple in estimator.transformers_:
        trfName, trf, trfCols = trfTuple
        componentScores.append(trf.score(X, y))
    return np.mean(componentScores)


def genReconstructionScorer(scoreFun):
    def scorer(estimator, X, y=None):
        feat = estimator.transform(X)
        rec = np.dot(feat, estimator.components_) + estimator.mean_
        return scoreFun(X, rec)
    return scorer


def reconstructionR2(estimator, X, y=None):
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            XInter = Pipeline(estimator.steps[:-1]).transform(X)
        else:
            XInter = X
        finalStep = estimator.steps[-1][1]
        feat = finalStep.transform(XInter)
        rec = np.dot(feat, finalStep.components_) + finalStep.mean_
        return r2_score(XInter, rec)
    else:
        feat = estimator.transform(X)
        rec = np.dot(feat, estimator.components_) + estimator.mean_
        return r2_score(X, rec)

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


class absMinMaxScaler(MinMaxScaler):

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            super().fit(X.abs(), y=y)
        else:
            super().fit(np.abs(X), y=y)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return super().transform(X.abs())
        else:
            return super().transform(np.abs(X))

class flatStandardScaler(StandardScaler):

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.means_per_column_ = X.mean()
            super().fit(X.to_numpy().reshape(-1, 1), y=y)
        else:
            self.means_per_column_ = np.mean(X)
            super().fit(X.reshape(-1, 1), y=y)
        return self

    def transform(self, X):
        originalShape = X.shape
        if isinstance(X, pd.DataFrame):
            res = super().transform(
                (X - self.means_per_column_).to_numpy().reshape(-1, 1))
        else:
            res = super().transform((X - self.means_per_column_).reshape(-1, 1))
        if isinstance(X, pd.DataFrame):
            output = pd.DataFrame(
                np.reshape(res, originalShape),
                index=X.index, columns=X.columns)
        else:
            output = np.reshape(res, originalShape)
        '''if True:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(X.xs(1, level='conditionUID').iloc[:, 0].to_numpy())
            ax[1].plot(output.xs(1, level='conditionUID').iloc[:, 0].to_numpy())
            plt.show()'''
        return output

class DataFrameStandardScaler(StandardScaler):

    def transform(self, X):
        res = super().transform(X)
        if isinstance(X, pd.DataFrame):
            output = pd.DataFrame(
                res,
                index=X.index, columns=X.columns)
        else:
            output = res
        '''if True:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(X.xs(1, level='conditionUID').iloc[:, 0].to_numpy())
            ax[1].plot(output.xs(1, level='conditionUID').iloc[:, 0].to_numpy())
            plt.show()'''
        return output


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
    # x is time
    # c is an offset
    # dc is the spacing parameter
    argCos = (x - c) * np.pi / dc / 2
    argCos[argCos > np.pi] = np.pi
    argCos[argCos < - np.pi] = - np.pi
    return (np.cos(argCos) + 1) / 2


def raisedCosBoundary(
        b=None, DT=None, minX=None, causal=True,
        nb=None, plotting=True, nlin=None, invnl=None):
    if nlin is None:
        nlin = lambda x: np.log(x + eps)
    if invnl is None:
        invnl = lambda x: np.exp(x) - eps
    boundsT = nlin(np.asarray([minX + b, minX + b + DT]))
    # print('boundsT {}'.format(boundsT))
    if causal:
        spacingT = (boundsT[1] - boundsT[0]) / (nb + 3)
        endpoints = invnl(np.asarray([boundsT[0] + 2 * spacingT, boundsT[1] - 2 * spacingT]))
    else:
        spacingT = (boundsT[1] - boundsT[0]) / (nb + 1)
        endpoints = invnl(np.asarray([boundsT[0], boundsT[1] - 2 * spacingT]))
    # print('endpoints {}'.format(endpoints))
    return endpoints - b

def makeBSplineBasis(
        nb=None, dt=None, endpoints=None,
        useFirst=False, useLast=False,
        normalize=False, causal=False):
    ihbasis, iht = None, None
    iht = np.arange(*endpoints, dt)
    numDFs = nb + 2 + (-1 if useFirst else 0) + (-1 if useLast else 0)
    ihbasis = bs(iht, df=numDFs, include_intercept=True)
    ctrs = np.linspace(*endpoints, numDFs+2)[1:-1]
    #
    keepMask = np.ones(ihbasis.shape[1], dtype=bool)
    keepMask[0] = useFirst
    keepMask[-1] = useLast
    #
    ihbasis = ihbasis[:, keepMask]
    ctrs = ctrs[keepMask]
    if normalize:
        for colIdx in range(ihbasis.shape[1]):
            ihbasis[:, colIdx] = ihbasis[:, colIdx] / np.sum(ihbasis[:, colIdx])
    iht = np.round(iht, decimals=6)
    if causal:
        ihbasis[iht <= 0] = 0
    orthobas = scipy.linalg.orth(ihbasis)
    ctrCols = np.round(ctrs, decimals=3)
    ihbDF = pd.DataFrame(ihbasis, index=iht, columns=ctrCols)
    orthobasDF = pd.DataFrame(orthobas, index=iht, columns=ctrCols)
    if causal:
        ihbDF = ihbDF.loc[iht >= 0, :]
        orthobasDF = orthobasDF.loc[iht >= 0, :]
    if iht[0] != 0:
        paddingT = np.round(np.arange(0, iht[0], dt), decimals=6)
        assert paddingT[-1] < iht[0]
        if paddingT.shape[0] > 0:
            paddingDF = pd.DataFrame(0, index=paddingT, columns=ctrCols)
            ihbDF = pd.concat([paddingDF, ihbDF])
            orthobasDF = pd.concat([paddingDF, orthobasDF])
    return ihbDF, orthobasDF

def makeRaisedCosBasis(
        nb=None, spacing=None, dt=None, endpoints=None, normalize=False, causal=False):
    """
        Make linearly stretched basis consisting of raised cosines
    """
    if spacing is None:
        spacing = (endpoints[1] - endpoints[0]) / (nb - 1)
    else:
        assert (nb is None)
    # centers for basis vectors
    '''ctrs = np.round(np.arange(
        endpoints[0], endpoints[1] + spacing, spacing), decimals=6)'''
    ctrs = np.arange(endpoints[0], endpoints[1] + spacing, spacing)
    if ctrs.size > nb:
        ctrs = ctrs[:nb]
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
    iht = np.round(iht, decimals=6)
    if causal:
        ihbasis[iht <= 0] = 0
    orthobas = scipy.linalg.orth(ihbasis)
    ihbDF = pd.DataFrame(ihbasis, index=iht, columns=np.round(ctrs, decimals=3))
    orthobasDF = pd.DataFrame(orthobas, index=iht, columns=np.round(ctrs, decimals=3))
    if causal:
        ihbDF = ihbDF.loc[iht >= 0, :]
        orthobasDF = orthobasDF.loc[iht >= 0, :]
    return ihbDF, orthobasDF


def makeLogRaisedCosBasis(
        nb, dt, endpoints, b=0.01,
        zflag=False, normalize=False,
        nlin=None, invnl=None, causal=False):
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
    if nlin is None:
        nlin = lambda x: np.log(x + eps)
    if invnl is None:
        invnl = lambda x: np.exp(x) - eps
    assert b > 0
    if isinstance(endpoints, list):
        endpoints = np.array(endpoints)
    #
    yrnge = nlin(endpoints + b)
    db = np.diff(yrnge)/(nb-1)  # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)  # centers for basis vectors
    if ctrs.size > nb:
        ctrs = ctrs[:nb]
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
    if normalize:
        for colIdx in range(ihbasis.shape[1]):
            ihbasis[:, colIdx] = ihbasis[:, colIdx] / np.sum(ihbasis[:, colIdx])
    if causal:
        ihbasis[iht <= 0] = 0
    orthobas = scipy.linalg.orth(ihbasis)
    # orthobas, _ = scipy.linalg.qr(ihbasis)
    ihbDF = pd.DataFrame(ihbasis, index=iht, columns=np.round(ctrs, decimals=3))
    orthobasDF = pd.DataFrame(orthobas, index=iht, columns=np.round(ctrs, decimals=3))
    if causal:
        ihbDF = ihbDF.loc[iht >= 0, :]
        orthobasDF = orthobasDF.loc[iht >= 0, :]
    #
    # negiht = np.sort(iht[iht > 0] * (-1))
    # negBDF = pd.DataFrame(0, index=negiht, columns=ihbDF.columns)
    # negOrthobasDF = pd.DataFrame(0, index=negiht, columns=orthobasDF.columns)
    # return pd.concat([negBDF, ihbDF]), pd.concat([negOrthobasDF, orthobasDF])
    return ihbDF, orthobasDF


def _poisson_pseudoR2(y, yhat):
    ynull = np.mean(y)
    yhat = yhat.reshape(y.shape)
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


class trialAwareStratifiedKFoldBackup:
    def __init__(
            self, n_splits=5, shuffle=False, random_state=None,
            stratifyFactors=None, continuousFactors=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratifyFactors = stratifyFactors
        self.continuousFactors = continuousFactors

    def split(self, X, y=None, groups=None):
        #
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
            #
            testCG = continuousDF['continuousGroup'].iloc[te]
            testMask = trialInfo['continuousGroup'].isin(testCG)
            testIdx = trialInfo.loc[testMask, :].index.to_list()
            folds.append((trainIdx, testIdx))
        return folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class trainTestValidationSplitterBackup:
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
        originalIndices = np.arange(dataDF.index.shape[0])[workIdx]
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

    def plot_schema(self):
        fig, ax = plt.subplots()
        for foldIdx, (tr, te) in enumerate(self.folds):
            lhtr, = ax.plot(tr, [foldIdx for j in tr], 'bo', alpha=0.3, label='train')
            lhte, = ax.plot(te, [foldIdx for j in te], 'ro', alpha=0.3, label='test')
        lhw, = ax.plot(self.work, [foldIdx + 1 for j in self.work], 'co', alpha=0.3, label='work')
        lhv, = ax.plot(self.validation, [foldIdx + 1 for j in self.validation], 'mo', alpha=0.3, label='validation')
        ax.legend(handles=[lhtr, lhte, lhw, lhv])
        return fig, ax


class EmpiricalCovarianceTransformer(EmpiricalCovariance, TransformerMixin):
    def __init__(self, maxNSamples=None, **kwds):
        self.maxNSamples = int(maxNSamples) if maxNSamples is not None else None
        super().__init__(**kwds)

    def fit(self, X, y=None):
        if self.maxNSamples is None:
            super().fit(X, y=y)
        else:
            # subsample X
            rng = np.random.default_rng()
            chooseN = min(self.maxNSamples, X.shape[0])
            if isinstance(X, pd.DataFrame):
                seekIdx = rng.choice(X.index, chooseN, replace=False)
                super().fit(X.loc[seekIdx, :], y=y)
            else:
                seekIdx = rng.choice(range(X.shape[0]), chooseN, replace=False)
                super().fit(X[seekIdx, :], y=y)
        return self

    def transform(self, X):
        return np.reshape(np.sqrt(self.mahalanobis(X)), (-1, 1))

class MinCovDetTransformer(MinCovDet, TransformerMixin):
    def __init__(self, maxNSamples=None, **kwds):
        self.maxNSamples = int(maxNSamples) if maxNSamples is not None else None
        super().__init__(**kwds)

    def fit(self, X, y=None):
        if self.maxNSamples is None:
            super().fit(X, y=y)
        else:
            # subsample X
            rng = np.random.default_rng()
            chooseN = min(self.maxNSamples, X.shape[0])
            if isinstance(X, pd.DataFrame):
                seekIdx = rng.choice(X.index, chooseN, replace=False)
                super().fit(X.loc[seekIdx, :], y=y)
            else:
                seekIdx = rng.choice(range(X.shape[0]), chooseN, replace=False)
                super().fit(X[seekIdx, :], y=y)
        return self

    def transform(self, X):
        return np.reshape(np.sqrt(self.mahalanobis(X)), (-1, 1))

class LedoitWolfTransformer(LedoitWolf, TransformerMixin):
    def __init__(self, maxNSamples=None, **kwds):
        self.maxNSamples = int(maxNSamples) if maxNSamples is not None else None
        self.rescaler = PowerTransformer(method='box-cox')
        super().__init__(**kwds)

    def fit(self, X, y=None):
        if self.maxNSamples is None:
            super().fit(X, y=y)
        else:
            # subsample X
            rng = np.random.default_rng()
            chooseN = min(self.maxNSamples, X.shape[0])
            if isinstance(X, pd.DataFrame):
                seekIdx = rng.choice(X.index, chooseN, replace=False)
                super().fit(X.loc[seekIdx, :], y=y)
                self.rescaler.fit(self.mahalanobis(X.loc[seekIdx, :]).reshape(-1, 1))
            else:
                seekIdx = rng.choice(range(X.shape[0]), chooseN, replace=False)
                super().fit(X[seekIdx, :], y=y)
                self.rescaler.fit(self.mahalanobis(X[seekIdx, :]).reshape(-1, 1))
        return self

    def transform(self, X):
        return self.rescaler.transform(self.mahalanobis(X).reshape(-1, 1))
        # return np.reshape(np.sqrt(self.mahalanobis(X)), (-1, 1))


class SMWrapper(BaseEstimator, RegressorMixin):
    """
        A universal sklearn-style wrapper for statsmodels regressors
        based on https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible/
        by David Dale
    """
    def __init__(
            self, sm_class, family=None,
            alpha=None, L1_wt=None, refit=None,
            maxiter=100, tol=1e-6, disp=False, check_step=True,
            fit_intercept=False, start_params=None, active_set=True,
            calc_frequency_weights=False, frequency_weights_index='trialUID',
            frequency_group_columns=[
                'electrode', 'trialAmplitude', 'trialRateInHz',
                'pedalMovementCat', 'pedalDirection', 'pedalSizeCat']
            ):
        self.sm_class = sm_class
        self.family = family
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.refit = refit
        self.active_set = active_set
        self.maxiter = maxiter
        self.tol = tol
        self.disp = disp
        self.check_step = check_step
        self.start_params = start_params
        self.calc_frequency_weights = calc_frequency_weights
        self.frequency_weights_index = frequency_weights_index
        self.frequency_group_columns = frequency_group_columns
        self.model_ = None
        self.results_ = None

    #
    def fit(self, X, y):
        model_opts = {}
        for key in dir(self):
            if key in ['family']:
                model_opts.update({key: getattr(self, key)})
        if self.calc_frequency_weights:
            infoByTrial = X.index.to_frame().drop_duplicates(self.frequency_weights_index).reset_index(drop=True)
            # dropCols = [cN for cN in ['conditionUID', 'bin'] if cN in infoByTrial.columns]
            # infoByTrial.drop(columns=dropCols, inplace=True)
            infoByTrial.loc[:, 'freqWeight'] = 1
            conditionCount = infoByTrial.groupby(self.frequency_group_columns).sum()['freqWeight']
            maxCount = conditionCount.max()
            for name, group in infoByTrial.groupby(self.frequency_group_columns):
                thisDeficit = group.shape[0]
                whole = maxCount // thisDeficit
                if whole > 0:
                    infoByTrial.loc[group.index, 'freqWeight'] += whole
                remainder = maxCount % thisDeficit
                if remainder > 0:
                    infoByTrial.loc[group.index[:remainder], 'freqWeight'] += 1
            weightMap = infoByTrial.loc[:, [self.frequency_weights_index, 'freqWeight']].set_index(self.frequency_weights_index)['freqWeight']
            frequency_weights = X.index.get_level_values(self.frequency_weights_index).map(weightMap).to_list()
            model_opts['freq_weights'] = frequency_weights
            # print('Setting frequency weights to : {}'.format(model_opts['freq_weights']))
        if self.fit_intercept:
            XX = sm.add_constant(X)
        else:
            XX = X
        try:
            self.model_ = self.sm_class(
                y, XX, **model_opts)
        except Exception:
            traceback.print_exc()
        ####
        if self.start_params is not None:
            if isinstance(self.start_params, float):
                self.start_params = self.start_params * np.ones(XX.shape[1])
            elif self.start_params == 'rand':
                rngNorm = np.random.default_rng()
                self.start_params = rngNorm.standard_normal(XX.shape[1])
        regular_opts = {}
        for key in dir(self):
            if key in ['alpha', 'L1_wt', 'refit', 'check_step', 'start_params', 'active_set']:
                if getattr(self, key) is not None:
                    regular_opts.update({key: getattr(self, key)})
        fit_opts = {}
        for key in dir(self):
            if key in ['maxiter', 'tol', 'disp']:
                if getattr(self, key) is not None:
                    fit_opts.update({key: getattr(self, key)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        self.coef_ = self.results_.params
        # print(self.results_.fit_history)
        self.n_iter_ = self.results_.fit_history['iteration']
        self.convergence_history = pd.DataFrame(
            self.results_.fit_history.pop('convergence_history', None),
            columns=['iter', 'param', 'cost'])
        self.summary_ = self.results_.summary()
        self.summary2_ = self.results_.summary2()
        self.results_.remove_data()
    #
    def predict(self, X):
        if self.fit_intercept:
            XX = sm.add_constant(X)
        else:
            XX = X
        return self.results_.predict(XX)
    #
    def score(self, X, y=None):
        if 'family' in dir(self):
            if 'Poisson' in str(self.family):
                return poisson_pseudoR2(self, X, y)
            if 'Gaussian' in str(self.family):
                return r2_score(y, self.predict(X))


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