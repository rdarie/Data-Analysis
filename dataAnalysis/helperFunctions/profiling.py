import psutil
import os


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def get_memory_usage():
    # return the memory usage in MB
    if 'psutil_process' in globals():
        psutil_process = globals()['psutil_process']
    else:
        psutil_process = psutil.Process(os.getpid())
        globals().update({'psutil_process': psutil_process})
    mem = psutil_process.memory_info()[0] / float(2 ** 20)
    return mem


def print_memory_usage(prefix='profiling', placeholder=None):
    if placeholder is None:
        placeholder = ': using {:.1f} MB'
    mem = get_memory_usage()
    print(prefix + placeholder.format(mem))
    return
