from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

ext_modules = (
    cythonize(["kalman_ode_higher.pyx", 
              "kalman_ode_higher.py"],
              annotate = True)
)
setup(
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]
)
