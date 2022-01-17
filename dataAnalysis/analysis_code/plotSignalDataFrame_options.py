import dataAnalysis.plotting.aligned_signal_plots as asp
import seaborn as sns
import pandas as pd
import pdb
minNObservations = 3

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

def shadeAUCEpochsPerFacet(g, ro, co, hu, dataSubset):
    emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
    commonPatchOpts = {
        'alpha': 0.15, 'zorder': -100}
    if not hasattr(g.axes[ro, co], 'aucEpochsShaded') and not emptySubset:
        colName = g.col_names[co]
        rowName = g.row_names[ro]
        g.axes[ro, co].axvspan(
            -100e-3, 200e-3, facecolor=iteratorPalette.loc['B', 'color'],
            **commonPatchOpts)
        g.axes[ro, co].aucEpochsShaded = True
        return

####
argumentsLookup = {
    'rig_illustration': {
        'recalcStats': True,
        'winStart': '-350', 'winStop': '1200', 'limitPages': None,
        'unitQuery': "chanName.isin(['position#0', 'position_x#0', 'position_y#0','velocity#0', 'velocity_x#0', 'velocity_y#0', 'amplitude#0', 'amplitude_raster#0', 'ins_td0#0', 'ins_td2#0'])",
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
        'winStart': '-150', 'winStop': '350', 'limitPages': None,
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
    'laplace_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '1000', 'limitPages': None,
        # 'unitQuery': None,
        'unitQuery': "chanName.isin(['utah_csd_14#0', 'utah_csd_30#0', 'utah_csd_33#0', 'utah_csd_3#0', 'utah_csd_2#0', 'utah_csd_79#0', 'utah_csd_60#0', 'utah_csd_80#0', 'utah_csd_16#0', 'utah_csd_17#0'])",
        # 'alignQuery': None,
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
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_illustration_topo': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '400', 'limitPages': None,
        'unitQuery': None,
        # 'alignQuery': "(trialRateInHz == 100.) & (pedalMovementCat == 'outbound')",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'yCoords', 'rowControl': '',
        'colName': 'xCoords', 'colControl': '',
        'groupPagesByColumn': 'all',
        'groupPagesByIndex': 'stimCondition, kinematicConditionNoSize',
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'laplace_spectral_illustration_topo': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '800', 'limitPages': None,
        'unitQuery': None,
        # 'alignQuery': "(trialRateInHz == 100.) & (pedalMovementCat == 'outbound')",
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
        'winStart': '-200', 'winStop': '350', 'limitPages': None,
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
        'winStart': '-200', 'winStop': '400', 'limitPages': None,
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
    'laplace_spectral_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '1000', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah_csd_2_alpha#0', 'utah_csd_60_alpha#0', 'utah_csd_14_alpha#0', 'utah_csd_63_alpha#0', 'utah_csd_2_beta#0', 'utah_csd_11_beta#0', 'utah_csd_56_beta#0', 'utah_csd_19_beta#0', 'utah_csd_30_gamma#0', 'utah_csd_60_gamma#0', 'utah_csd_53_gamma#0', 'utah_csd_17_gamma#0', 'utah_csd_40_higamma#0', 'utah_csd_60_higamma#0', 'utah_csd_27_higamma#0', 'utah_csd_14_higamma#0', 'utah_csd_70_spb#0', 'utah_csd_80_spb#0', 'utah_csd_28_spb#0', 'utah_csd_14_spb#0'])",
        # 'unitQuery': None,
        # 'alignQuery': None,
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
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'mahal_illustration': {
        'recalcStats': True,
        'winStart': '-200', 'winStop': '400', 'limitPages': None,
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
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'mahal_per_trial_illustration': {
        'recalcStats': True,
        'winStart': '-250', 'winStop': '350', 'limitPages': None,
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
        'winStart': '-250', 'winStop': '350', 'limitPages': None,
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
    'laplace_illustration_topo': dict(
        referenceTimeWindow=None,
        # referenceTimeWindow=[-400e-3, -350e-3],
        testStride=100e-3,
        testWidth=100e-3,
        tStart=-200e-3,
        tStop=None,
        pThresh=5e-2,
        correctMultiple=True,
        ),
    'laplace_spectral_illustration_topo': dict(
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
        fbPrettyName = '\n({})'.format(titleLabelLookup[fbn]) if fbn is not None else ''
        titleLabelLookup['feature = utah_csd_{}{}'.format(eIdx, fbSuffix)] = 'LFP chan. #{}{}'.format(eIdx, fbPrettyName)
        titleLabelLookup['utah_csd_{}{}'.format(eIdx, fbSuffix)] = 'LFP chan. #{}{}'.format(eIdx, fbPrettyName)
plotProcFunsLookup = {
    'rig_illustration': [
    # shadeTaskEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'lfp_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=-100e-3, tOffset=.4,
            includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup)
        ],
    'laplace_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=-100e-3, tOffset=.4, includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup)
        ],
    'laplace_illustration_topo': [
        # shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'laplace_spectral_illustration_topo': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genLegendRounder(decimals=2),
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        # asp.genTitleChanger(titleLabelLookup)
        ],
    'factor_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}, dropNaNCol='trialUID'),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=0., tOffset=.35, includeLeft=False, includeRight=False, dropNaNCol='trialUID'),
        asp.genTitleChanger(titleLabelLookup)],
    'spectral_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=-.10, tOffset=.4, includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup)],
    'laplace_spectral_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=-0.10, tOffset=.4, includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup)],
    'mahal_illustration': [
        shadeAUCEpochsPerFacet,
        asp.xLabelsTime,
        asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
        asp.genLegendRounder(decimals=2),
        asp.genStimVLineAdder(
            'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
            tOnset=-.10, tOffset=.4, includeLeft=False, includeRight=False),
        asp.genTitleChanger(titleLabelLookup),
            # asp.genTraceAnnotator(
            #     unit_var='trialUID', labelsList=['segment', 't'],
            #     textOpts=dict(ha='left', va='bottom', fontsize=4))
            ]
    }
plotProcFunsLookup['mahal_per_trial_illustration'] = plotProcFunsLookup['mahal_illustration']
plotProcFunsLookup['rig_per_trial_illustration'] = plotProcFunsLookup['rig_illustration']

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
    'lfp_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'laplace_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'laplace_illustration_topo': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True},
        'errorbar': 'se'
    },
    'laplace_spectral_illustration_topo': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'facet_kws': {'sharey': True},
        'errorbar': 'se'
    },
    'factor_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'spectral_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'laplace_spectral_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:0.,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
    },
    'mahal_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1",
        'errorbar': 'se'
        # 'facet_kws': {'sharey': False},
        # 'estimator': None, 'units': 'trialUID'
    },
    'mahal_per_trial_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:-0.8,-.3,dark=.25,light=0.75,reverse=1",
        'alpha': 0.5,
        'errorbar': None
    },
    'rig_per_trial_illustration': {
        'linewidth': .75, 'height': 2, 'aspect': 1.5,
        'palette': "ch:0.8,-.3,dark=.25,light=0.75,reverse=1",
        'alpha': 0.5,
        'errorbar': None
    }
}

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
    'laplace_illustration_topo': {
        'linewidth': 1., 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.25,light=0.75,reverse=1"
    },
    'laplace_spectral_illustration_topo': {
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
    'laplace_illustration_topo': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'laplace_spectral_illustration_topo': {
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
    'laplace_illustration_topo': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'laplace_spectral_illustration_topo': {
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
    'mahal_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    }
}

styleOptsLookup['mahal_per_trial_illustration'] = styleOptsLookup['mahal_illustration']
styleOptsLookup['rig_per_trial_illustration'] = styleOptsLookup['rig_illustration']
xAxisLabelLookup = {
    }
yAxisLabelLookup = {
    'rig_illustration': {
        'amplitude': 'Stimulation amplitude (uA)',
        'position': 'Pedal position (deg.)',
    },
    'lfp_illustration': {
        'utah18': 'LFP recording (uV)',
    },
    'factor_illustration': {
        'utah18': 'LFP recording (uV)',
    },
    'spectral_illustration': {
        'utah18_alpha': 'LFP alpha (uV)',
        'utah18_beta': 'LFP beta (uV)',
        'utah18_gamma': 'LFP gamma (uV)',
        'utah18_higamma': 'LFP high gamma (uV)',
        'utah18_spb': 'LFP SPB (uV)',
    },
    'mahal_illustration': {
        'mahal_ledoit_all': 'Mahalanobis\ndistance (a.u.)',
    }
}
yAxisLabelLookup['mahal_per_trial_illustration'] = yAxisLabelLookup['mahal_illustration']
yAxisLabelLookup['rig_per_trial_illustration'] = yAxisLabelLookup['rig_illustration']
titlesOptsLookup = {
    'mahal_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'rig_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_illustration_topo': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_illustration_topo': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'laplace_spectral_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    }

titlesOptsLookup['mahal_per_trial_illustration'] = titlesOptsLookup['mahal_illustration']
titlesOptsLookup['rig_per_trial_illustration'] = titlesOptsLookup['rig_illustration']
titleTextLookup = {}
customCodeLookup = {
    'rig_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100",
    'rig_per_trial_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100",
    'laplace_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'laplace_spectral_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()",
    'mahal_illustration': "arguments['colOrder'] = trialInfo.drop_duplicates().sort_values(['trialRateInHz', 'electrode'])['stimCondition'].unique().tolist()"
    }