import psutil
import time
import os, sys
import numpy as np
import collections
import line_profiler
import pdb
from inspect import getmembers, isfunction

psutil_process = psutil.Process(os.getpid())
startTime = 0


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def get_memory_usage():
    # return the memory usage in MB
    # if 'psutil_process' in globals():
    #     psutil_process = globals()['psutil_process']
    # else:
    #     psutil_process = psutil.Process(os.getpid())
    #     globals().update({'psutil_process': psutil_process})
    global psutil_process
    mem = psutil_process.memory_info()[0] / float(2 ** 20)
    return mem


def print_memory_usage(prefix='profiling', placeholder=None):
    if placeholder is None:
        placeholder = ': using {:.1f} MB'
    mem = get_memory_usage()
    print(prefix + placeholder.format(mem))
    return


def start_timing(mess='Starting timer...'):
    global startTime
    startTime = time.perf_counter()
    return


def stop_timing(prefix='profiling', placeholder=None):
    if placeholder is None:
        placeholder = ': took {:.1f} sec'
    global startTime
    endTime = time.perf_counter()
    print(prefix + placeholder.format(endTime - startTime))
    return


def register_module_with_profiler(mod, profile):
    functionList = [o[1] for o in getmembers(mod) if isfunction(o[1])]
    for thisFun in functionList:
        profile.add_function(thisFun)
    return


def register_list_with_profiler(functionList, profile):
    for thisFun in functionList:
        if isfunction(thisFun):
            profile.add_function(thisFun)
    return


#  hack original line_profiler.show_text to override dict order
def show_profiler_text(stats, unit, output_unit=None, stream=None, stripzeros=False):
    """ Show text for the given timings.
    """
    if stream is None:
        stream = sys.stdout

    if output_unit is not None:
        stream.write('Timer unit: %g s\n\n' % output_unit)
    else:
        stream.write('Timer unit: %g s\n\n' % unit)

    for (fn, lineno, name), timings in stats.items():
        line_profiler.show_func(
            fn, lineno, name, stats[fn, lineno, name], unit,
            output_unit=output_unit, stream=stream, stripzeros=stripzeros)
    return


def orderLStatsByTime(scriptName):
    statsPackage = line_profiler.load_stats(scriptName)
    stats = {k: v for k, v in statsPackage.timings.items() if len(v)}
    unit = statsPackage.unit
    allKeys = []
    totalTimes = []
    for (fn, lineno, name), timings in stats.items():
        total_time = 0.0
        for inner_lineno, nhits, thisTime in timings:
            total_time += thisTime
        allKeys.append((fn, lineno, name))
        totalTimes.append(total_time)
    orderedIdx = np.argsort(totalTimes)
    orderedStats = collections.OrderedDict()
    for i in orderedIdx:
        orderedStats[allKeys[i]] = stats[allKeys[i]]
    return orderedStats, unit


def profileFunction(
        topFun=None, modulesToProfile=None,
        registerTopFun=True,
        outputBaseFolder='.', nameSuffix=''):
    if not os.path.exists(outputBaseFolder):
        os.makedirs(outputBaseFolder, exist_ok=True)
    profile = line_profiler.LineProfiler()
    for mod in modulesToProfile:
        register_module_with_profiler(mod, profile)
    if registerTopFun:
        profile.add_function(topFun)
    #
    profile.runcall(topFun)
    #
    if nameSuffix is not None:
        fileName = os.path.basename(__file__) + '_' + nameSuffix
    else:
        fileName = os.path.basename(__file__)
    outfile = os.path.join(
        outputBaseFolder,
        '{}.{}'.format(fileName, 'lprof'))
    profile.dump_stats(outfile)
    orderedStats, unit = orderLStatsByTime(outfile)
    outfiletext = os.path.join(
        outputBaseFolder,
        '{}.{}'.format(fileName, 'lprof.txt'))
    with open(outfiletext, 'w') as f:
        show_profiler_text(orderedStats, unit, stream=f)
    return
