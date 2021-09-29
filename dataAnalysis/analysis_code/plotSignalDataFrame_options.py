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
        'winStart': '-250', 'winStop': '500', 'limitPages': None,
        'unitQuery': "chanName.isin(['position#0', 'amplitude#0'])", 'alignQuery': "conditionUID == 2",
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimConditionWithDate', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        # 'colOrder': [
        #     'NA_NA', 'CW_outbound', 'CW_return',
        #     'CCW_outbound', 'CCW_return'
        #     ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'lfp_illustration': {
        'recalcStats': True,
        'winStart': '-250', 'winStop': '500', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah_csd_81#0', 'utah_csd_18#0'])", 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimConditionWithDate', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        #  'colOrder': [
        #      'NA_NA', 'CW_outbound', 'CW_return',
        #      'CCW_outbound', 'CCW_return'
        #      ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'factor_illustration': {
        'recalcStats': True,
        'winStart': '-250', 'winStop': '500', 'limitPages': None,
        'unitQuery': "chanName.isin(['fa_all001#0', 'fa_all002#0', 'fa_all003#0', 'fa_all004#0'])", 'alignQuery': None,
        # 'unitQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA', 'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return'
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'spectral_illustration': {
        'recalcStats': True,
        'winStart': '-250', 'winStop': '500', 'limitPages': None,
        'unitQuery': "chanName.isin(['utah_csd_81_alpha#0', 'utah_csd_81_beta#0', 'utah_csd_81_gamma#0', 'utah_csd_81_higamma#0', 'utah_csd_81_spb#0'])",
        # 'unitQuery': None,
        'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA', 'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return'
            ],
        'sizeName': '', 'sizeControl': '',
        'styleName': '', 'styleControl': ''},
    'mahal_illustration': {
        'recalcStats': True,
        'winStart': '-250', 'winStop': '500', 'limitPages': None,
        'unitQuery': None, 'alignQuery': None,
        'individualTraces': False, 'overlayStats': False,
        'hueName': 'trialAmplitude', 'hueControl': '',
        'rowName': 'stimCondition', 'rowControl': '', # 'rowOrder':
        'colName': 'kinematicConditionNoSize', 'colControl': '',
        'colOrder': [
            'NA_NA', 'CW_outbound', 'CW_return',
            'CCW_outbound', 'CCW_return'
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

plotProcFunsLookup = {
    'rig_illustration': [
    shadeTaskEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
        # asp.genTitleChanger({
        #     'stimCondition = + E16 - E5_100.0': '+ E16 - E5 (100.0 Hz)',
        #     'stimCondition = NA_0.0': 'No stimulation',
        #     'kinematicConditionNoSize = NA_NA': 'No movement',
        #     'kinematicConditionNoSize = CW_outbound': 'Start of movement (extension)',
        #     'kinematicConditionNoSize = CW_return': 'Return to start (flexion)'
        #     })
        ],
    'lfp_illustration': [
    shadeAUCEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
    asp.genStimVLineAdder(
        'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
        tOnset=-.05, tOffset=.7, includeLeft=False, includeRight=False),
        # asp.genTitleChanger({
        #     'stimCondition = + E16 - E5_100.0': '+ E16 - E5 (100.0 Hz)',
        #     'stimCondition = NA_0.0': 'No stimulation',
        #     'kinematicConditionNoSize = NA_NA': 'No movement',
        #     'kinematicConditionNoSize = CW_outbound': 'Start of movement (extension)',
        #     'kinematicConditionNoSize = CW_return': 'Return to start (flexion)',
        #     'kinematicConditionNoSize = CCW_outbound': 'Start of movement (flexion)',
        #     'kinematicConditionNoSize = CCW_return': 'Return to start (extension)',
        #     })
        ],
    'factor_illustration': [
    shadeAUCEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}, dropNaNCol='trialUID'),
    asp.genLegendRounder(decimals=2),
    asp.genStimVLineAdder(
        'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
        tOnset=-.05, tOffset=.7, includeLeft=False, includeRight=False, dropNaNCol='trialUID'),
    asp.genTitleChanger({
        'stimCondition = + E16 - E5_100.0': '+ E16 - E5 (100.0 Hz)',
        'stimCondition = NA_0.0': 'No stimulation',
        'kinematicConditionNoSize = NA_NA': 'No movement',
        'kinematicConditionNoSize = CW_outbound': 'Start of movement (extension)',
        'kinematicConditionNoSize = CW_return': 'Return to start (flexion)',
        'kinematicConditionNoSize = CCW_outbound': 'Start of movement (flexion)',
        'kinematicConditionNoSize = CCW_return': 'Return to start (extension)',
        })],
    'spectral_illustration': [
    shadeAUCEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
    asp.genStimVLineAdder(
        'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
        tOnset=-.05, tOffset=.7, includeLeft=False, includeRight=False),
    asp.genTitleChanger({
        'stimCondition = + E16 - E5_100.0': '+ E16 - E5 (100.0 Hz)',
        'stimCondition = NA_0.0': 'No stimulation',
        'kinematicConditionNoSize = NA_NA': 'No movement',
        'kinematicConditionNoSize = CW_outbound': 'Start of movement (extension)',
        'kinematicConditionNoSize = CW_return': 'Return to start (flexion)',
        'kinematicConditionNoSize = CCW_outbound': 'Start of movement (flexion)',
        'kinematicConditionNoSize = CCW_return': 'Return to start (extension)',
        })],
    'mahal_illustration': [
    shadeAUCEpochsPerFacet,
    asp.xLabelsTime,
    asp.genVLineAdder([0], {'color': 'y', 'alpha': 0.5}),
    asp.genLegendRounder(decimals=2),
    asp.genStimVLineAdder(
        'trialRateInHz', {'color': 'y', 'lw': 0.5, 'alpha': 1, 'ymin': 0.9, 'ymax': .95},
        tOnset=-.05, tOffset=.7, includeLeft=False, includeRight=False),
    asp.genTitleChanger({
        'stimCondition = + E16 - E5_100.0': '+ E16 - E5 (100.0 Hz)',
        'stimCondition = NA_0.0': 'No stimulation',
        'kinematicConditionNoSize = NA_NA': 'No movement',
        'kinematicConditionNoSize = CW_outbound': 'Start of movement (extension)',
        'kinematicConditionNoSize = CW_return': 'Return to start (flexion)',
        'kinematicConditionNoSize = CCW_outbound': 'Start of movement (flexion)',
        'kinematicConditionNoSize = CCW_return': 'Return to start (extension)',
        })]
    }
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
        # for evoked lfp report
        # xUnitFactor=1e3, yUnitFactor=1,
        # xUnits='msec', yUnits='uV',
        # for evoked emg report
        xUnitFactor=1e3, yUnitFactor=1,
        xUnits='msec', yUnits='uV',
    ),
    asp.genTraceAnnotator(
        unit_var='trialUID', labelsList=['segment', 't'],
        textOpts=dict(ha='left', va='bottom', fontsize=4))
]
relPlotKWArgsLookup = {
    'rig_illustration': {
        'linewidth': 1, 'height': 4, 'aspect': 2,
        'palette': "ch:0.8,-.3,dark=.1,light=0.7,reverse=1"
    },
    'lfp_illustration': {
        'linewidth': .5, 'height': 4, 'aspect': 2,
        'palette': "ch:1.6,-.3,dark=.1,light=0.7,reverse=1"
    },
    'factor_illustration': {
        'linewidth': .5, 'height': 1.5, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.1,light=0.7,reverse=1"
    },
    'spectral_illustration': {
        'linewidth': .5, 'height': 1.5, 'aspect': 3,
        'palette': "ch:0.,-.3,dark=.1,light=0.7,reverse=1"
    },
    'mahal_illustration': {
        'linewidth': .5, 'height': 1.5, 'aspect': 3,
        'palette': "ch:-0.8,-.3,dark=.1,light=0.7,reverse=1"
    }
}

catPlotKWArgsLookup = {
    'rig_illustration': {
        'linewidth': .5, 'height': 1.5, 'aspect': 3,
        'palette': "ch:0.8,-.3,dark=.1,light=0.7,reverse=1"
    },
    'lfp_illustration': {
        'linewidth': .5, 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.1,light=0.7,reverse=1"
    },
    'factor_illustration': {
        'linewidth': .5, 'height': 1, 'aspect': 3,
        'palette': "ch:1.6,-.3,dark=.1,light=0.7,reverse=1"
    },
    'spectral_illustration': {
        'linewidth': .5, 'height': 1, 'aspect': 3,
        'palette': "ch:0.,-.3,dark=.1,light=0.7,reverse=1"
    },
    'mahal_illustration': {
        'linewidth': .5, 'height': 1, 'aspect': 3,
        'palette': "ch:-0.8,-.3,dark=.1,light=0.7,reverse=1"
    }
}
legendTitleOverridesLookup = {
    'rig_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'lfp_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'factor_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    },
    'spectral_illustration': {
        'trialAmplitude': 'Stimulation\namplitude (uA)',
    }
}
legendContentOverridesLookup = {}
styleOptsLookup = {
    'rig_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'lfp_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'factor_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    },
    'spectral_illustration': {
        'legend.lw': 2, 'tight_layout.pad': 2e-1,
    }
}
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
titlesOptsLookup = {
    'rig_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    'lfp_illustration': {'col_template': '{col_name}', 'row_template': '{row_name}'},
    }
titleTextLookup = {}
customCodeLookup = {
    'rig_illustration': "dataDF.loc[:, idxSl['position', :, :, :, :]] = dataDF.loc[:, idxSl['position', :, :, :, :]] * 100"
    }