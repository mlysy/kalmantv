from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import scipy as sp

ext_modules = (
    cythonize("mat_mult.pyx")
)
setup(
    ext_modules = ext_modules,
    include_dirs=[np.get_include(), sp.get_include()]
)
