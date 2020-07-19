from setuptools import setup, find_packages, Extension
import numpy as np
import scipy as sp
# from setuptools import setup, find_packages
from os import path

eigen_path = "eigen-3.3.7"
# compile with cython if it's installed
try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

cmdclass = {}
if USE_CYTHON:
    # extensions = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})


# c/cpp modules
ext_c = '.pyx' if USE_CYTHON else '.c'
ext_cpp = '.pyx' if USE_CYTHON else 'cpp'
ext_modules = [Extension("kalmantv.tests.kalmantv_eigen",
                         ["kalmantv"+ext_cpp],
                         include_dirs=[
                             np.get_include(),
                             eigen_path],
                         extra_compile_args=['-O3'],
                         language='c++')]
setup(
    name='kalmantv_test',
    version='0.0.2',
    author='Mohan Wu, Martin Lysy',
    author_email='mlysy@uwaterloo.ca',
    description='Eigen backed Kalman Filtering and Smoothing for testing',
    url="https://github.com/mlysy/probDE",
    packages=[],

    # cython
    cmdclass=cmdclass,
    ext_modules=ext_modules,

    install_requires=['numpy', 'scipy', 'matplotlib'],
    setup_requires=['setuptools>=38'],

    # install_requires=['numpy', 'scipy', 'matplotlib']
    # packages=['probDE', 'probDE/Bayesian', 'probDE/Kalman', 'probDE/Kalman/Old', 'probDE/Kalman/pykalman', 'probDE/utils', 'probDE/Tests']
)
