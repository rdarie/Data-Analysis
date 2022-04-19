import dataAnalysis.plotting.aligned_signal_plots as asp
import seaborn as sns
import pandas as pd
import numpy as np
import pdb
from itertools import product
minNObservations = 3
exampleChannelStr = 'utah_csd_2'
colorMaps = {
    # coldish
    'covMat': 'crest_r',
    'points': 'mako',
    # warmish
    'distMat': 'flare_r',
    'distHist': 'rocket'
    }
iteratorDescriptions = pd.Series({
    'cd': 'B',
    'cb': 'M',
    'ccs': 'S',
    'ccm': 'SM',
    })
rawIteratorColors = sns.color_palette(colorMaps['points'], iteratorDescriptions.shape[0])
iteratorPalette = pd.DataFrame(
    rawIteratorColors, columns=['r', 'g', 'b'],
    index=iteratorDescriptions.to_list()).apply(lambda x: tuple(x), axis='columns').to_frame(name='color')
##### custom functions

def shadeTaskEpochsPerFacet(g, ro, co, hu, dataSubset):
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
    commonPatchOpts = {
        'alpha': 0.15, 'zorder': -100}
    if not hasattr(g.axes[ro, co], 'taskEpochsShaded') and not emptySubset:
        colName = g.col_names[co]
        rowName = g.row_names[ro]
        # baseline
        if ('outbound' in colName):
            g.axes[ro, co].axvspan(
                -700e-3, -400e-3, facecolor=iteratorPalette.loc['B', 'color'],
                **commonPatchOpts)
        ### active period
        if (colName == 'NA_NA') and ('NA' not in rowName):
            # stim only
            g.axes[ro, co].axvspan(
                -200e-3, 600e-3, facecolor=iteratorPalette.loc['S', 'color'],
                **commonPatchOpts)
        elif ('NA' in rowName):
            # movement only
            g.axes[ro, co].axvspan(
                -200e-3, 600e-3, facecolor=iteratorPalette.loc['M', 'color'],
                **commonPatchOpts)
        else:
            # movement and stim
            g.axes[ro, co].axvspan(
                -200e-3, 600e-3, facecolor=iteratorPalette.loc['SM', 'color'],
                **commonPatchOpts)
        g.axes[ro, co].taskEpochsShaded = True
        return
#
exampleLaplaceChannelListStr = ', '.join([
    "'utah_csd_{}{}#0'".format(cn, fb) for fb, cn in product(
        ['', '_alpha', '_beta', '_gamma', '_higamma', '_spb'],
        [2, 8] # 2, 8, 17, 35
    )])
exampleLfpChannelListStr = ', '.join([
    "'utah{}{}#0'".format(cn, fb) for fb, cn in product(
        ['', '_alpha', '_beta', '_gamma', '_higamma', '_spb'],
        [90, 96, 86, 64])])
#
argumentsLookup = {
    'rig_illustration': {
        'recalcStats': True,
        'winStart': '-800', 'winStop': '1000', 'limitPages': None,
        'unitQuery': "chanName.isin(['position#0','velocity_x#0','velocity_y#0', 'amplitude#0', 'utah_rawAverage_0#0', 'ins_td0#0', 'ins_td2#0'])",
        # 'alignQuery': None,
        # 'alignQuery': "conditionUID == 2",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimConditionWithDate', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'lfp_illustration': {
        'recalcStats': True,
        'winStart': '-25', 'winStop': '60', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah1#0', 'utah75#0'])",
        # 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'rig_hf_illustration': {
        'recalcStats': True,
        'winStart': '-5', 'winStop': '35', 'limitPages': None,
        'unitQuery': "chanName.isin(['position#0','velocity_x#0','velocity_y#0', 'amplitude#0', 'utah_rawAverage_0#0', 'ins_td0#0', 'ins_td2#0'])",
        # 'alignQuery': None,
        'alignQuery': "((trialRateInHz == 50.) | (trialRateInHz == 100.))",
        # 'alignQuery': "conditionUID == 2",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'lfp_hf_illustration': {
        'recalcStats': True,
        'winStart': '-5', 'winStop': '35', 'limitPages': None,
        'unitQuery': "chanName.isin([" + exampleLfpChannelListStr + ", 'averageSignal#0'])",
        'alignQuery': "((trialRateInHz == 50.) | (trialRateInHz == 100.))",
        # 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_hf_illustration': {
        'recalcStats': True,
        'winStart': '-5', 'winStop': '35', 'limitPages': None,
        'unitQuery': "chanName.isin([" + exampleLaplaceChannelListStr + "])",
        'alignQuery': "((trialRateInHz == 50.) | (trialRateInHz == 100.))",
        # 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '',
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_hf_topo_illustration': {
        'recalcStats': True,
        'winStart': '-5', 'winStop': '35', 'limitPages': None,
        'alignQuery': "((trialRateInHz == 50.) | (trialRateInHz == 100.))",
        # 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'yCoords', 'rowControl': '',
        'colName': 'xCoords', 'colControl': '',
        'groupPagesByColumn': 'all',
        'groupPagesByIndex': 'stimCondition, kinematicConditionNoSize',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_topo_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '225', 'limitPages': None,
        'unitQuery': None,
        'alignQuery': "((trialRateInHz == 0.) | (trialRateInHz == 100.)) & (pedalMovementCat == 'outbound')",
        # 'alignQuery': "(electrode == 'NA') & (pedalMovementCat == 'outbound')",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'yCoords', 'rowControl': '',
        'colName': 'xCoords', 'colControl': '',
        'groupPagesByColumn': 'all',
        'groupPagesByIndex': 'stimCondition, kinematicConditionNoSize',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'lfp_illustration_topo': {
        'recalcStats': True,
        'winStart': '-20', 'winStop': '40', 'limitPages': None,
        'unitQuery': None,
        'alignQuery': "(trialRateInHz == 100.) & (pedalMovementCat == 'outbound')",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'yCoords', 'rowControl': '',
        'colName': 'xCoords', 'colControl': '',
        'groupPagesByColumn': 'all',
        'groupPagesByIndex': 'stimCondition, kinematicConditionNoSize',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_spectral_topo_illustration': {
        'recalcStats': True,
        'winStart': '-125', 'winStop': '225', 'limitPages': None,
        'unitQuery': None,
        # 'alignQuery': "((trialRateInHz == 0.) | (trialRateInHz == 100.)) & (pedalMovementCat == 'outbound')",
        'alignQuery': "(trialRateInHz == 0.) & (pedalMovementCat == 'outbound')",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'yCoords', 'rowControl': '',
        'colName': 'xCoords', 'colControl': '',
        'groupPagesByColumn': 'freqBandName',
        'groupPagesByIndex': 'stimCondition, kinematicConditionNoSize',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'factor_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '300', 'limitPages': None,
        'unitQuery': "chanName.isin(['fa_all001#0', 'fa_all002#0', 'fa_all003#0', 'fa_all004#0'])",
        # 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'spectral_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '225', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah1_alpha#0', 'utah1_beta#0', 'utah1_gamma#0', 'utah1_higamma#0', 'utah1_spb#0'])",
        # 'unitQuery': None,
        # 'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_illustration': {
        'recalcStats': True,
        'winStart': '-300', 'winStop': '225', 'limitPages': None,
        # 'unitQuery': None,
        'unitQuery': "chanName.isin([" + exampleLaplaceChannelListStr + "])", # 'unitQuery': "chanName.isin(['{}#0', 'utah_csd_8#0', 'utah_csd_35#0', 'utah_csd_17#0',])".format(exampleChannelStr), 'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'colName': 'stimCondition', 'colControl': '',  # 'rowOrder':
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'kinematicConditionNoSize', 'rowControl': '',
        'rowOrder': [
           'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        # 'groupPagesByColumn': 'feature',
        # 'groupPagesByIndex': 'stimCondition',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_auc_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '225', 'limitPages': None,
        # 'unitQuery': None,
        'unitQuery': "chanName.isin([" + exampleLaplaceChannelListStr + "])", # 'unitQuery': "chanName.isin(['{}#0', 'utah_csd_8#0', 'utah_csd_35#0', 'utah_csd_17#0',])".format(exampleChannelStr), 'alignQuery': None,
        # 'alignQuery': "(trialRateInHz.isin([100.])) & (pedalMovementCat == 'outbound')",
        'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'colName': 'stimCondition', 'colControl': '',
        # 'rowOrder':
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'kinematicConditionNoSize', 'rowControl': '',
        'rowOrder': [
           'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        # 'groupPagesByColumn': 'feature',
        # 'groupPagesByIndex': 'stimCondition',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_spectral_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '225', 'limitPages': None,
        'unitQuery': "chanName.isin([" + exampleLaplaceChannelListStr + "])",
        'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        #
        'colName': 'stimCondition', 'colControl': '',
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'kinematicConditionNoSize', 'rowControl': '',
        'rowOrder': [
           'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        # 'groupPagesByColumn': 'feature',
        # 'groupPagesByIndex': 'stimCondition',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'mahal_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '225', 'limitPages': None,
        'unitQuery': None, 'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'colName': 'stimCondition', 'colControl': '',
        # 'rowOrder':
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'kinematicConditionNoSize', 'rowControl': '',
        'rowOrder': [
           'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        # 'groupPagesByColumn': 'feature',
        # 'groupPagesByIndex': 'stimCondition',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'mahal_per_trial_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '300', 'limitPages': None,
        'unitQuery': None, 'alignQuery': None,
        'individualTraces': True, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'rig_per_trial_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '300', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah_artifact_0#0', 'utah_rawAverage_0#0', 'position#0'])",
        # 'unitQuery': None, 'alignQuery': None,
        'individualTraces': True, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA',
            'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return',
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''}
    }
argumentsLookup['laplace_aucNoStim_illustration'] = argumentsLookup['laplace_auc_illustration'].copy()
argumentsLookup['laplace_aucNoStim_illustration']['groupPagesByColumn'] = 'feature'
argumentsLookup['laplace_aucNoStim_illustration']['groupPagesByIndex'] = 'stimCondition'
argumentsLookup['laplace_aucNoStim_illustration']['alignQuery'] = "(trialRateInHz == 0.)"
#"(trialRateInHz == 0.) & (pedalMovementCat == 'outbound')"
#
argumentsLookup['laplace_spectral_auc_illustration'] = argumentsLookup['laplace_spectral_illustration'].copy()
argumentsLookup['laplace_spectral_auc_illustration']['groupPagesByColumn'] = 'parentFeature'
argumentsLookup['laplace_spectral_auc_illustration']['groupPagesByIndex'] = 'stimCondition'
argumentsLookup['laplace_spectral_auc_illustration']['colName'] = 'feature'
#
argumentsLookup['laplace_spectral_aucNoStim_illustration'] = argumentsLookup['laplace_spectral_auc_illustration'].copy()
argumentsLookup['laplace_spectral_aucNoStim_illustration']['groupPagesByColumn'] = 'parentFeature'
argumentsLookup['laplace_spectral_aucNoStim_illustration']['groupPagesByIndex'] = 'stimCondition'
argumentsLookup['laplace_spectral_aucNoStim_illustration']['colName'] = 'feature'
argumentsLookup['laplace_spectral_aucNoStim_illustration']['alignQuery'] = "(trialRateInHz == 0.)"
#"(trialRateInHz == 0.) & (pedalMovementCat == 'outbound')"
#
argumentsLookup['mahal_auc_illustration'] = argumentsLookup['mahal_illustration'].copy()
argumentsLookup['mahal_auc_illustration']['unitQuery'] = None
#
argumentsLookup['mahal_aucNoStim_illustration'] = argumentsLookup['mahal_auc_illustration'].copy()
argumentsLookup['mahal_aucNoStim_illustration']['groupPagesByColumn'] = 'feature'
argumentsLookup['mahal_aucNoStim_illustration']['groupPagesByIndex'] = 'stimCondition'
argumentsLookup['mahal_aucNoStim_illustration']['alignQuery'] = 'trialRateInHz == 0.'
#
argumentsLookup['laplace_std_illustration'] = argumentsLookup['laplace_illustration'].copy()
argumentsLookup['lfp_hf_topo_illustration'] = argumentsLookup['laplace_hf_topo_illustration'].copy()
#
argumentsLookup['laplace_spectral_std_topo_illustration'] = argumentsLookup['laplace_spectral_topo_illustration'].copy()

statsTestOpts = dict(
    referenceTimeWindow=None,
    # referenceTimeWindow=[-400e-3, -350e-3],
    testStride=100e-3,
    testWidth=100e-3,
    tStart=-200e-3,
    tStop=None,
    pThresh=5e-2,
    correctMultiple=True,
    )

statsTestOptsLookup = {
    'rig_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'lfp_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'laplace_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'lfp_illustration_topo': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'laplace_topo_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'laplace_spectral_topo_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'factor_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'spectral_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'laplace_spectral_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'mahal_illustration': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    }
statsTestOptsLookup['mahal_per_trial_illustration'] = statsTestOptsLookup['mahal_illustration']
statsTestOptsLookup['rig_per_trial_illustration'] = statsTestOptsLookup['rig_illustration']
statsTestOptsLookup['rig_hf_illustration'] = statsTestOptsLookup['rig_illustration']
statsTestOptsLookup['lfp_hf_illustration'] = statsTestOptsLookup['lfp_illustration']
statsTestOptsLookup['laplace_hf_illustration'] = statsTestOptsLookup['laplace_illustration']
statsTestOptsLookup['laplace_auc_illustration'] = statsTestOptsLookup['laplace_illustration']
statsTestOptsLookup['mahal_auc_illustration'] = statsTestOptsLookup['laplace_illustration']
statsTestOptsLookup['laplace_spectral_auc_illustration'] = statsTestOptsLookup['laplace_spectral_illustration'].copy()
statsTestOptsLookup['laplace_aucNoStim_illustration'] = statsTestOptsLookup['laplace_illustration']
statsTestOptsLookup['mahal_aucNoStim_illustration'] = statsTestOptsLookup['mahal_illustration']
statsTestOptsLookup['laplace_spectral_aucNoStim_illustration'] = statsTestOptsLookup['laplace_spectral_illustration'].copy()
statsTestOptsLookup['laplace_std_illustration'] = statsTestOptsLookup['laplace_illustration'].copy()
statsTestOptsLookup['laplace_hf_topo_illustration'] = statsTestOptsLookup['laplace_illustration'].copy()
statsTestOptsLookup['laplace_spectral_std_topo_illustration'] = statsTestOptsLookup['laplace_spectral_topo_illustration'].copy()
statsTestOptsLookup['lfp_hf_topo_illustration'] = statsTestOptsLookup['laplace_illustration'].copy()

titleLabelLookup = {
    'stimCondition = NA_0.0': 'No stim.',
    'NA_0.0': 'No stim.',
    'kinematicCondition = NA_NA': 'No movement',
    'kinematicCondition = CW_outbound': 'Start of movement\n(extension)',
    'kinematicCondition = CW_return': 'Return to start\n(flexion)',
    'kinematicCondition = CCW_outbound': 'Start of movement\n(flexion)',
    'kinematicCondition = CCW_return': 'Return to start\n(extension)',
    'kinematicConditionNoSize = NA_NA': 'No movement',
    'kinematicConditionNoSize = CW_outbound': 'Start of movement\n(extension)',
    'kinematicConditionNoSize = CW_return': 'Return to start\n(flexion)',
    'kinematicConditionNoSize = CCW_outbound': 'Start of movement\n(flexion)',
    'kinematicConditionNoSize = CCW_return': 'Return to start\n(extension)',
    'NA_NA': 'No movement',
    'CW_outbound': 'Start of movement\n(extension)',
    'CW_return': 'Return to start\n(flexion)',
    'CCW_outbound': 'Start of movement\n(flexion)',
    'CCW_return': 'Return to start\n(extension)',
    'broadband': 'Broadband',
    'alpha': 'Alpha',
    'beta': 'Beta',
    'gamma': 'Gamma',
    'higamma': 'High gamma',
    'spb': 'Spike band',
    }
for eIdx in range(16):
    titleLabelLookup['stimCondition = -E{:02d}+E16_100.0'.format(eIdx)] = 'Stim. E{:02d} (100 Hz)'.format(eIdx)
    titleLabelLookup['stimCondition = -E{:02d}+E16_50.0'.format(eIdx)] = 'Stim. E{:02d} (50 Hz)'.format(eIdx)
    titleLabelLookup['-E{:02d}+E16_100.0'.format(eIdx)] = 'Stim. E{:02d} (100 Hz)'.format(eIdx)
    titleLabelLookup['-E{:02d}+E16_50.0'.format(eIdx)] = 'Stim. E{:02d} (50 Hz)'.format(eIdx)
for eIdx in range(97):
    for fbn in ['alpha', 'beta', 'gamma', 'higamma', 'spb', None]:
        fbSuffix = '_{}'.format(fbn) if fbn is not None else ''
        # fbPrettyName = '\n({})'.format(titleLabelLookup[fbn]) if fbn is not None else ''
        fbPrettyName = '{} '.format(titleLabelLookup[fbn]) if fbn is not None else ''
        titleLabelLookup['feature = utah_csd_{}{}'.format(eIdx, fbSuffix)] = '{}LFP ch. #{}'.format(fbPrettyName,eIdx)
        titleLabelLookup['utah_csd_{}{}'.format(eIdx, fbSuffix)] = '{}LFP ch. #{}'.format(fbPrettyName, eIdx)

plotProcFunsLookup = {
    'rig_illustration': [
    # shadeTaskEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'rig_hf_illustration': [
    # shadeTaskEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'lfp_illustration': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'y', 'lw': 1.5, 'alpha': 1,
                'ymin': 0.9, 'ymax': .95},
            tOnset=0., tOffset=60e-3,
            includeLeft=True, includeRight=False),
        asp.genTitleChanger(titleLabelLookup),
        asp.genAxisLabelOverride(
            xTemplate=None, yTemplate='{feature}',
            titleTemplate=None, colKeys=['feature'])
        ],
    'lfp_hf_illustration': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'y', 'lw': 1.5, 'alpha': 1,
                'ymin': 1.01, 'ymax': 1.04},
            tOnset=0., tOffset=80e-3,
            delayMap={
                'kinematicConditionNoSize': {
                    'NA_NA': 0.,
                    'CW_outbound': 0.,
                    'CCW_outbound': 0.,
                    'CW_return': 0.,
                    'CCW_return': 0.}
                },
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genPedalPosAdder(
            autoscale=True,
            yMin=1.06, yMax=1.14,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genAxisLabelOverride(
            xTemplate=None, yTemplate='{feature}',
            titleTemplate=None, colKeys=['feature'])
        ],
    'laplace_hf_illustration': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        # asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'y', 'lw': 1.5, 'alpha': 1,
                'ymin': 1.01, 'ymax': 1.04},
            tOnset=0., tOffset=80e-3,
            delayMap={
                'kinematicConditionNoSize': {
                    'NA_NA': 0.,
                    'CW_outbound': 0.,
                    'CCW_outbound': 0.,
                    'CW_return': 0.,
                    'CCW_return': 0.,
                    }
                },
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genPedalPosAdder(
            autoscale=True,
            yMin=1.06, yMax=1.14,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genAxisLabelOverride(
            xTemplate=None, yTemplate='{feature}',
            titleTemplate=None, colKeys=['feature'])
        ],
    'laplace_topo_illustration': [
        # shadeAUCEpochsPerFacet,
        # asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genXLimSetter(quantileLims=1),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genHLineAdder([0], {'color': '.1', 'alpha': 0.5, 'ls': '--'}),
        asp.genTicksToScale(
            lineOpts={'lw': 1}, shared=True,
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            ),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'lfp_illustration_topo': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'laplace_spectral_topo_illustration': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genXLimSetter(quantileLims=1),
        asp.genTicksToScale(
            lineOpts={'lw': 1}, shared=True,
            xUnitFactor=1e3, yUnitFactor=1,
            xUnits='msec', yUnits='uV',
            ),
        asp.genVLineAdder([0], {'color': 'c', 'alpha': 0.5}),
        asp.genHLineAdder([0], {'color': '.1', 'alpha': 0.5, 'ls': '--'}),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'factor_illustration': [
        #shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}, dropNaNCol='trialUID'),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=0., tOffset=.35, includeLeft=False, includeRight=False, dropNaNCol='trialUID'),
        asp.genTitleChanger(titleLabelLookup)],
    'spectral_illustration': [
        #shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'c', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'c', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=0., tOffset=.4, includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup)],
    'laplace_illustration': [
        asp.genAUCEpochShader(
            span=(-100e-3, 200e-3),
            commonPatchOpts={'alpha': 0.15, 'zorder': -100, 'facecolor': iteratorPalette.loc['B', 'color']}),
        asp.genGridFormatter(),
        asp.genXLimSetter(quantileLims=1),
        # asp.genAxTickRemover(y=True),
        asp.genVLineAdder([0], {'color': 'm', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'y', 'lw': .5, 'alpha': 1,
                'ymin': 0.96, 'ymax': 0.99},
            tOnset=0., tOffset=250e-3,
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genPedalPosAdder(
            autoscale=True,
            yMin=0.86, yMax=0.94,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        ],
    'laplace_auc_illustration': [
        asp.genAUCEpochShader(
            span=(-100e-3, 200e-3),
            commonPatchOpts={
                'alpha': 0.05, 'zorder': -100,
                # 'linewidth': 0.5, 'edgecolor': iteratorPalette.loc['B', 'color'],
                'facecolor': iteratorPalette.loc['B', 'color']}),
        asp.genAUCShader(span=(-100e-3, 200e-3), patchOpts=dict(alpha=0.5)),
        asp.genGridFormatter(),
        # asp.genAxTickRemover(y=True),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genHLineAdder([0], {'color': '.1', 'alpha': 0.5, 'ls': '--'}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        #
        asp.genXLimSetter(quantileLims=1),
        asp.genAxLimShiftScale(yScale=0.86 ** (-1)),  # matches 0.86 lower bound of annotations
        #
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'y', 'lw': .5, 'alpha': 1,
                'ymin': 0.96, 'ymax': 0.99},
            tOnset=0., tOffset=250e-3,
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(
                x=-125e-3, fontweight='bold', va='center')),
        asp.genPedalPosAdder(
            autoscale=True, tStart=-125e-3,
            yMin=0.86, yMax=0.94,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(
                fontweight='bold', va='center')),
        ],
    'laplace_spectral_illustration': [
        asp.genAUCEpochShader(
            span=(-100e-3, 200e-3),
            commonPatchOpts={'alpha': 0.15, 'zorder': -100, 'facecolor': iteratorPalette.loc['B', 'color']}),
        asp.genGridFormatter(),
        asp.genXLimSetter(quantileLims=1),
        # asp.genAxTickRemover(y=True),
        asp.genVLineAdder([0], {'color': 'b', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'c', 'lw': .5, 'alpha': 1,
                'ymin': 0.96, 'ymax': 0.99},
            tOnset=0., tOffset=250e-3,
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genPedalPosAdder(
            autoscale=True,
            yMin=0.84, yMax=0.94,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        ],
    'mahal_illustration': [
        asp.genAUCEpochShader(
            span=(-100e-3, 200e-3),
            commonPatchOpts={'alpha': 0.15, 'zorder': -100, 'facecolor': iteratorPalette.loc['B', 'color']}),
        asp.genGridFormatter(),
        asp.genXLimSetter(quantileLims=1),
        # asp.genAxTickRemover(y=True),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genTitleChanger(titleLabelLookup),
        asp.genStimVLineAdder(
            'trialRateInHz', {
                'color': 'g', 'lw': .5, 'alpha': 1,
                'ymin': 0.96, 'ymax': 0.99},
            tOnset=0., tOffset=250e-3,
            includeLeft=True, includeRight=False,
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
        asp.genPedalPosAdder(
            autoscale=True,
            yMin=0.86, yMax=0.94,
            plotOptsMain=dict(lw=.5, c='k'),
            plotOptsBounds=dict(lw=.5, c='k', alpha=0.5, ls='--'),
            addTitle=True, titleFontOpts=dict(fontweight='bold')),
            ]
    }
plotProcFunsLookup['mahal_per_trial_illustration'] = plotProcFunsLookup['mahal_illustration']
plotProcFunsLookup['rig_per_trial_illustration'] = plotProcFunsLookup['rig_illustration']

plotProcFunsLookup['laplace_aucNoStim_illustration'] = plotProcFunsLookup['laplace_auc_illustration'].copy()
#
plotProcFunsLookup['laplace_spectral_auc_illustration'] = plotProcFunsLookup['laplace_auc_illustration'].copy()
plotProcFunsLookup['laplace_spectral_aucNoStim_illustration'] = plotProcFunsLookup['laplace_auc_illustration'].copy()

plotProcFunsLookup['mahal_auc_illustration'] = plotProcFunsLookup['laplace_auc_illustration'].copy()
plotProcFunsLookup['mahal_aucNoStim_illustration'] = plotProcFunsLookup['laplace_auc_illustration'].copy()

plotProcFunsLookup['laplace_std_illustration'] = plotProcFunsLookup['laplace_illustration'].copy()
plotProcFunsLookup['laplace_hf_topo_illustration'] = plotProcFunsLookup['laplace_illustration'].copy()
plotProcFunsLookup['laplace_spectral_std_topo_illustration'] = plotProcFunsLookup['laplace_spectral_topo_illustration'].copy()
plotProcFunsLookup['laplace_spectral_std_topo_illustration'][-1] = asp.genHLineAdder([1], {'color': '.1', 'alpha': 0.5, 'ls': '--'})
plotProcFunsLookup['lfp_hf_topo_illustration'] = plotProcFunsLookup['laplace_illustration'].copy()

unusedPlotProcFuns = [
    asp.genNumRepAnnotator(
        hue_var=argumentsLookup['rig_illustration']['hueName'],
        unit_var='trialUID',
        xpos=0.05, ypos=.95, textOpts=dict(
            ha='left', va='top',
            c=(0., 0., 0., 0.7),
            bbox=dict(
                boxstyle="square",
                ec=(0., 0., 0., 0.),
                fc=(1., 1., 1., 0.2))
        )),
        asp.genGridAnnotator(
            xpos=.9, ypos=.9, template='{}',
            colNames=['feature'],
            textOpts={
                'fontsize': 10,
                'verticalalignment': 'top',
                'horizontalalignment': 'right'
            }, shared=False),
    # asp.genBlockVertShader([
    #         max(0e-3, alignedAsigsKWargs['windowSize'][0]),
    #         min(.9e-3, alignedAsigsKWargs['windowSize'][1])],
    #     asigPlotShadingOpts),
    # asp.genStimVLineAdder(
    #     'RateInHz', vLineOpts, tOnset=0, tOffset=.3, includeRight=False),
    # asp.genYLimSetter(newLims=[-75, 100], forceLims=True),
    asp.genTicksToScale(
        lineOpts={'lw': 2}, shared=True,
        xUnitFactor=1e3, yUnitFactor=1,
        xUnits='msec', yUnits='uV',
        ),
    asp.genTraceAnnotator(
        unit_var='trialUID', labelsList=['segment', 't'],
        textOpts=dict(ha='left', va='bottom', fontsize=4))
    ]
relPlotKWArgsLookup = {
    'rig_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:0.8,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'rig_hf_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 3,
        'palette': "ch:0.8,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': None
    },
    'lfp_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'lfp_hf_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': None
    },
    'laplace_hf_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': None
    },
    'lfp_illustration_topo': {
        'linewidth': 2., 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': False},
        'errorbar': None  # 'se'
    },
    'laplace_topo_illustration': {
        'linewidth': 1.5, 'height': .8, 'aspect': 1,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True, 'sharex': True},
        'errorbar': None,
        'rasterized': False, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
    },
    'factor_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.2,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'laplace_illustration': {
        'linewidth': 1., 'height': 2, 'aspect': 1.2,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se',
        'rasterized': False, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
    },
    'laplace_auc_illustration': {
        'linewidth': 1., 'height': 2., 'aspect': 1.2,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True, 'sharex': True},
        'errorbar': None, 'rasterized': False, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
    },
    'laplace_spectral_illustration': {
        'linewidth': 1., 'height': 2, 'aspect': 1.2,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True, 'sharex': True},
        'errorbar': 'sd', 'rasterized': True, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
        },
    'spectral_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.2,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'laplace_spectral_topo_illustration': {
        'linewidth': 1.5, 'height': .8, 'aspect': 1,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True, 'sharex': True},
        'errorbar': None,
        'rasterized': False, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
    },
    'mahal_illustration': {
        'linewidth': 1., 'height': 2, 'aspect': 1.2,
        'facet_kws': {'sharey': True, 'sharex': True},
        'palette': "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se', 'rasterized': False, 'solid_joinstyle': 'round', 'solid_capstyle': 'round',
    },
    'mahal_per_trial_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1",
        'alpha': 0.5, 'errorbar': None,
    },
    'rig_per_trial_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:0.8,-.3,dark=.25,light=0.75,reverse=1",
        'alpha': 0.5,
        'errorbar': None
    }
    }
# 'alignQuery': ,
if argumentsLookup['laplace_spectral_topo_illustration']['alignQuery'] == "(trialRateInHz == 0.) & (pedalMovementCat == 'outbound')":
    relPlotKWArgsLookup['laplace_spectral_topo_illustration']['palette'] = "ch:-.3,-.01,dark=.25,light=0.26,reverse=1"
#
relPlotKWArgsLookup['laplace_hf_topo_illustration'] = relPlotKWArgsLookup['laplace_topo_illustration'].copy()
relPlotKWArgsLookup['laplace_spectral_std_topo_illustration'] = relPlotKWArgsLookup['laplace_spectral_topo_illustration'].copy()
relPlotKWArgsLookup['laplace_spectral_std_topo_illustration']['estimator'] = np.std
relPlotKWArgsLookup['laplace_spectral_std_topo_illustration']['errorbar'] = None
#
relPlotKWArgsLookup['lfp_hf_topo_illustration'] = relPlotKWArgsLookup['laplace_topo_illustration'].copy()
relPlotKWArgsLookup['lfp_hf_topo_illustration']['rasterized'] = True
relPlotKWArgsLookup['lfp_hf_topo_illustration']['aspect'] = 1.2
#
relPlotKWArgsLookup['laplace_spectral_auc_illustration'] = relPlotKWArgsLookup['laplace_auc_illustration'].copy()
relPlotKWArgsLookup['laplace_spectral_auc_illustration']['palette'] = relPlotKWArgsLookup['laplace_spectral_illustration']['palette']
relPlotKWArgsLookup['laplace_spectral_auc_illustration']['height'] = 2.
relPlotKWArgsLookup['laplace_spectral_auc_illustration']['aspect'] = 1.2
#
relPlotKWArgsLookup['mahal_auc_illustration'] = relPlotKWArgsLookup['laplace_auc_illustration'].copy()
relPlotKWArgsLookup['mahal_auc_illustration']['palette'] = relPlotKWArgsLookup['mahal_illustration']['palette']
relPlotKWArgsLookup['mahal_auc_illustration']['height'] = 2.
relPlotKWArgsLookup['mahal_auc_illustration']['aspect'] = 1.2
#
relPlotKWArgsLookup['laplace_aucNoStim_illustration'] = relPlotKWArgsLookup['laplace_auc_illustration'].copy()
relPlotKWArgsLookup['laplace_aucNoStim_illustration']['palette'] = "ch:1.3,-.1,dark=.25,light=0.26,reverse=1"
#
relPlotKWArgsLookup['mahal_aucNoStim_illustration'] = relPlotKWArgsLookup['mahal_auc_illustration'].copy()
relPlotKWArgsLookup['mahal_aucNoStim_illustration']['palette'] = "ch:-1.1,-.01,dark=.25,light=0.26,reverse=1"
#
relPlotKWArgsLookup['laplace_spectral_aucNoStim_illustration'] = relPlotKWArgsLookup['laplace_auc_illustration'].copy()
relPlotKWArgsLookup['laplace_spectral_aucNoStim_illustration']['palette'] = "ch:0.,-.3,dark=.25,light=0.26,reverse=1"
#
relPlotKWArgsLookup['laplace_std_illustration'] = relPlotKWArgsLookup['laplace_illustration'].copy()
relPlotKWArgsLookup['laplace_std_illustration']['estimator'] = np.std
relPlotKWArgsLookup['laplace_std_illustration']['errorbar'] = None
# relPlotKWArgsLookup['laplace_std_illustration']['palette'] = "ch:0.,-.2,dark=.25,light=0.75,reverse=1",
relPlotKWArgsLookup['laplace_std_illustration']['linewidth'] = 1.
#
catPlotKWArgsLookup = {
    'rig_illustration': {
        'linewidth': 1., 'height': 1.5, 'aspect': 3,
        'palette': "ch:0.8,-.3,dark=.25,light=0.75,reverse=1"
    },
    'lfp_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'laplace_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'lfp_illustration_topo': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'laplace_topo_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'laplace_spectral_topo_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'factor_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'spectral_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1"
    },
    'laplace_spectral_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1"
    },
    'mahal_illustration': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1"
    }
}
catPlotKWArgsLookup['mahal_per_trial_illustration'] = catPlotKWArgsLookup['mahal_illustration']
catPlotKWArgsLookup['rig_per_trial_illustration'] = catPlotKWArgsLookup['rig_illustration']
catPlotKWArgsLookup['rig_hf_illustration'] = catPlotKWArgsLookup['rig_illustration']
catPlotKWArgsLookup['lfp_hf_illustration'] = catPlotKWArgsLookup['lfp_illustration']
catPlotKWArgsLookup['laplace_hf_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['laplace_auc_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['mahal_auc_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['laplace_spectral_auc_illustration'] = catPlotKWArgsLookup['laplace_spectral_illustration']
catPlotKWArgsLookup['laplace_aucNoStim_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['mahal_aucNoStim_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['laplace_spectral_aucStim_illustration'] = catPlotKWArgsLookup['laplace_spectral_illustration']
catPlotKWArgsLookup['laplace_std_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['laplace_hf_topo_illustration'] = catPlotKWArgsLookup['laplace_illustration']
catPlotKWArgsLookup['laplace_spectral_std_topo_illustration'] = catPlotKWArgsLookup['laplace_spectral_topo_illustration']
catPlotKWArgsLookup['lfp_hf_topo_illustration'] = catPlotKWArgsLookup['laplace_illustration']
#
legendTitleOverridesLookup = {
    'rig_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'lfp_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'lfp_illustration_topo': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_topo_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_spectral_topo_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_spectral_std_topo_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'factor_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'spectral_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_spectral_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'mahal_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    }
}
legendTitleOverridesLookup['mahal_per_trial_illustration'] = legendTitleOverridesLookup['mahal_illustration']
legendTitleOverridesLookup['rig_per_trial_illustration'] = legendTitleOverridesLookup['rig_illustration']
legendTitleOverridesLookup['rig_hf_illustration'] = legendTitleOverridesLookup['rig_illustration']
legendTitleOverridesLookup['lfp_hf_illustration'] = legendTitleOverridesLookup['lfp_illustration']
legendTitleOverridesLookup['laplace_hf_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['laplace_auc_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['mahal_auc_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['laplace_spectral_auc_illustration'] = legendTitleOverridesLookup['laplace_spectral_illustration']
legendTitleOverridesLookup['laplace_aucNoStim_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['mahal_aucNoStim_illustration'] = legendTitleOverridesLookup['mahal_illustration']
legendTitleOverridesLookup['laplace_spectral_aucNoStim_illustration'] = legendTitleOverridesLookup['laplace_spectral_illustration']
legendTitleOverridesLookup['laplace_std_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['laplace_hf_topo_illustration'] = legendTitleOverridesLookup['laplace_illustration']
legendTitleOverridesLookup['laplace_spectral_topo_illustration'] = legendTitleOverridesLookup['laplace_spectral_topo_illustration']
legendTitleOverridesLookup['lfp_hf_topo_illustration'] = legendTitleOverridesLookup['laplace_illustration']

legendContentOverridesLookup = {}

styleOptsLookup = {
    'rig_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'lfp_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'lfp_illustration_topo': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_topo_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_spectral_topo_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_spectral_std_topo_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'factor_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'spectral_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_spectral_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_spectral_auc_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'mahal_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    }
}

styleOptsLookup['mahal_per_trial_illustration'] = styleOptsLookup['mahal_illustration']
styleOptsLookup['rig_per_trial_illustration'] = styleOptsLookup['rig_illustration']
styleOptsLookup['rig_hf_illustration'] = styleOptsLookup['rig_illustration']
styleOptsLookup['lfp_hf_illustration'] = styleOptsLookup['lfp_illustration']
styleOptsLookup['laplace_hf_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['laplace_auc_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['mahal_auc_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['laplace_spectral_auc_illustration'] = styleOptsLookup['laplace_spectral_illustration']
styleOptsLookup['laplace_aucNoStim_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['mahal_aucNoStim_illustration'] = styleOptsLookup['mahal_illustration']
styleOptsLookup['laplace_spectral_aucNoStim_illustration'] = styleOptsLookup['laplace_spectral_illustration']
styleOptsLookup['laplace_std_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['laplace_hf_topo_illustration'] = styleOptsLookup['laplace_illustration']
styleOptsLookup['laplace_spectral_std_topo_illustration'] = styleOptsLookup['laplace_spectral_topo_illustration']
styleOptsLookup['lfp_hf_topo_illustration'] = styleOptsLookup['laplace_illustration']
xAxisLabelLookup = {
    }
yAxisLabelLookup = {
    }
xAxisUnitsLookup = {
    }
yAxisUnitsLookup = {
    'laplace_illustration': '(z-scored, s.d.)',
    'laplace_auc_illustration': '(z-scored, s.d.)',
    'laplace_aucNoStim_illustration': '(z-scored, s.d.)',
    #
    'mahal_illustration': '(a.u.)',
    'mahal_auc_illustration': '(a.u.)',
    'mahal_aucNoStim_illustration': '(a.u.)',
    #
    'laplace_spectral_illustration': '(z-scored, s.d.)',
    'laplace_spectral_auc_illustration': '(z-scored, s.d.)',
    'laplace_spectral_aucNoStim_illustration': '(z-scored, s.d.)',
    #
    'laplace_std_illustration': '(std.dev., a.u.)'
    }

titlesOptsLookup = {
    'mahal_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'rig_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_hf_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_hf_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_illustration_topo': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_topo_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_std_topo_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_hf_topo_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_hf_topo_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_topo_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_auc_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_auc_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_aucNoStim_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_aucNoStim_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'mahal_aucNoStim_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_std_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    }
sharexAcrossPagesLookup = {
    }
shareyAcrossPagesLookup = {
    'mahal_illustration': False,
    'laplace_illustration': False,
    'laplace_spectral_illustration': False,
    #
    'laplace_auc_illustration': False,
    'laplace_aucNoStim_illustration': False,
    #
    'mahal_auc_illustration': False,
    'mahal_aucNoStim_illustration': False,
    #
    'laplace_spectral_auc_illustration': False,
    'laplace_spectral_aucNoStim_illustration': False,
    #
    'laplace_std_illustration': False,
    }
detrendLookup = {
    #
    'laplace_illustration': 'perTrial',
    'laplace_auc_illustration': 'perTrial',
    'laplace_aucNoStim_illustration': 'perTrial',
    #
    'mahal_illustration': 'perTrial',
    'mahal_auc_illustration': 'perTrial',
    'mahal_aucNoStim_illustration': 'perTrial',
    #
    'laplace_spectral_illustration': 'perTrial',
    'laplace_spectral_auc_illustration': 'perTrial',
    'laplace_spectral_aucNoStim_illustration': 'perTrial',
    #
    'laplace_std_illustration': 'noDetrend',
}
titlesOptsLookup['mahal_per_trial_illustration'] = titlesOptsLookup['mahal_illustration']
titlesOptsLookup['rig_per_trial_illustration'] = titlesOptsLookup['rig_illustration']
titlesOptsLookup['rig_hf_illustration'] = titlesOptsLookup['rig_illustration']
titlesOptsLookup['lfp_hf_illustration'] = titlesOptsLookup['lfp_illustration']
titlesOptsLookup['laplace_hf_topo_illustration'] = titlesOptsLookup['lfp_illustration']
titlesOptsLookup['laplace_spectral_std_topo_illustration'] = titlesOptsLookup['lfp_illustration']
titlesOptsLookup['lfp_hf_topo_illustration'] = titlesOptsLookup['lfp_illustration']
titlesOptsLookup['laplace_hf_illustration'] = titlesOptsLookup['laplace_illustration']
titlesOptsLookup['laplace_auc_illustration'] = titlesOptsLookup['laplace_illustration']
titlesOptsLookup['laplace_spectral_auc_illustration'] = titlesOptsLookup['laplace_spectral_illustration']
titlesOptsLookup['laplace_std_illustration'] = titlesOptsLookup['laplace_illustration']
titleTextLookup = {}

customCodeLookup = {
    'rig_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100",
    'rig_per_trial_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100",
    'rig_hf_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100",
    'laplace_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'laplace_std_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'laplace_aucNoStim_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'mahal_auc_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'mahal_aucNoStim_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'laplace_auc_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'laplace_spectral_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    # 'laplace_spectral_auc_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    # 'laplace_spectral_aucNoStim_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'mahal_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    }