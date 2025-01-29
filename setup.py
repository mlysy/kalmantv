from setuptools import setup, find_packages, Extension
import numpy as np
import platform
import os
import eigenpip as epip
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
docs_url = "https://kalmantv.readthedocs.io/en/latest/"

# readthedocs install
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# compile with cython if it's installed
try:
    # from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
# USE_CYTHON = False

# # compiler options
# if platform.system() != "Windows":
#     extra_compile_args = ["-O3", "-ffast-math",
#                           "-mtune=native", "-march=native"]
#     if platform.system() != "Darwin" and not on_rtd:
#         # default compiler on macOS doesn't support openmp
#         extra_compile_args.append("-fopenmp")
# else:
#     extra_compile_args = ["-O2", "/openmp"]
extra_compile_args = []

# remove numpy depreciation warnings as documented here:
# https://cython.readthedocs.io/en/latest/src/userguide/migrating_to_cy30.html#numpy-c-api
disable_numpy_warnings = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# cpp modules
ext_c = ".pyx" if USE_CYTHON else ".c"
ext_cpp = ".pyx" if USE_CYTHON else ".cpp"
ext_modules = [Extension("kalmantv.cython.blas",
                         ["kalmantv/cython/blas"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         # sp.get_include()],
                         extra_compile_args=extra_compile_args,
                         # extra_link_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c"),
               Extension("kalmantv.cython.kalmantv",
                         ["kalmantv/cython/kalmantv"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         extra_compile_args=extra_compile_args,
                         # extra_link_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c"),
               Extension("kalmantv.eigen.kalmantv",
                         ["kalmantv/eigen/kalmantv"+ext_cpp],
                         include_dirs=[
                             np.get_include(),
                             epip.get_include()],
                         extra_compile_args=extra_compile_args,
                         # extra_link_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c++"),
               Extension("kalmantv.eigen.omp_init",
                         ["kalmantv/eigen/omp_init"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         extra_compile_args=extra_compile_args,
                         # extra_link_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c")]


# cmdclass = {}
if USE_CYTHON:
    ext_modules = cythonize(ext_modules)
    # cmdclass.update({"build_ext": build_ext})


setup(
    name="kalmantv",
    version="0.2.5",
    author="Mohan Wu, Martin Lysy",
    author_email="mlysy@uwaterloo.ca",
    description="High-Performance Kalman Filtering and Smoothing",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="Kalman Cython",
    url="http://github.com/mlysy/kalmantv",
    project_urls={
        "Documentation": docs_url
    },
    packages=["kalmantv/cython", "kalmantv/numba", "kalmantv/eigen",
              "kalmantv",
              # "kalmantv/include/eigen"
              ],
    #package_dir={"kalmantv/include/eigen": EIGEN_PATH},
    package_data={
        "kalmantv/cython": ["*.pyx", "*.pxd"],
        "kalmantv/eigen": ["*.pyx", "*.pxd", "*.h"],
        # "kalmantv/include/eigen": extra_files
    },
    #package_data = packagefiles,
    # cython
    # cmdclass=cmdclass,
    ext_modules=ext_modules,
    python_requires='>=3.8',
    install_requires=[
        "cython>=3.0", "numpy>=1.22", 'scipy>=1.13',
        'numba>=0.51.2', 'eigenpip'
    ],
    extras_require={
        'docs': ['sphinx', 'sphinx_rtd_theme', 'recommonmark'],
        'tests': ['pandas', 'tox']
    }
)
