from setuptools import setup, find_packages

setup(
    name='ProprioDataAnalysis',
    version='0.1',
    packages=find_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires=[
        'numpy', 'dill', 'docopt', 'psutil', 'lmfit',
        'setuptools', 'mpi4py', 'line_profiler', 'tables',
        'statsmodels', 'pandas', 'libtfr', 'tzlocal', 'opencv_python', 
        'joblib', 'h5py', 'matplotlib', 'scipy', 'pyqtgraph', 'quantities', 
        'scikit-learn', 'tabulate'],
    long_description=open('README.txt').read(),
)
