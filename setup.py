from setuptools import setup, find_packages, Extension
import numpy as np
import platform
import os
# import scipy as sp

# compile with cython if it's installed
try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

cmdclass = {}
if USE_CYTHON:
    # extensions = cythonize(extensions)
    cmdclass.update({"build_ext": build_ext})

# path to eigen library
EIGEN_PATH = r"eigen-3.3.7"
def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files(EIGEN_PATH)
def write_eigen():
    cnt = """
# THIS FILE IS GENERATED FROM KALMANTV SETUP.PY
#
import kalmantv
import os
def get_include():
    return os.path.join(kalmantv.__path__[0], "include/eigen")
"""
    a = open("kalmantv/eigen_path.py", 'w')
    try:
        a.write(cnt)
    finally:
        a.close()

# compiler options
if platform.system() != "Windows":
    extra_compile_args = ["-O3", "-ffast-math",
                          "-mtune=native", "-march=native"]
    if platform.system() != "Darwin":
        # default compiler on macOS doesn't support openmp
        extra_compile_args.append("-fopenmp")
else:
    extra_compile_args = ["-O2","/openmp"]

# remove numpy depreciation warnings as documented here:
#
disable_numpy_warnings = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# cpp modules
ext_c = ".pyx" if USE_CYTHON else ".c"
ext_cpp = ".pyx" if USE_CYTHON else "cpp"
ext_modules = [Extension("kalmantv.cython.blas",
                         ["kalmantv/cython/blas"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         # sp.get_include()],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c"),
               Extension("kalmantv.cython.kalmantv",
                         ["kalmantv/cython/kalmantv"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c"),
               Extension("kalmantv.eigen.kalmantv",
                         ["kalmantv/eigen/kalmantv"+ext_cpp],
                         include_dirs=[
                             np.get_include(),
                             EIGEN_PATH],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c++"),
               Extension("kalmantv.eigen.omp_init",
                         ["kalmantv/eigen/omp_init"+ext_c],
                         include_dirs=[
                             np.get_include()],
                         extra_compile_args=extra_compile_args,
                         define_macros=disable_numpy_warnings,
                         language="c")]

write_eigen()
setup(
    name="kalmantv",
    version="0.2",
    author="Mohan Wu, Martin Lysy",
    author_email="mlysy@uwaterloo.ca",
    description="High-Performance Kalman Filtering and Smoothing",
    keywords="Kalman Cython",
    url="http://github.com/mlysy/kalmantv",
    packages=["kalmantv/cython", "kalmantv/numba", "kalmantv/eigen",
            "kalmantv", "kalmantv/include/eigen"],
    package_dir = {"kalmantv/include/eigen":EIGEN_PATH},
    package_data={
        "kalmantv/cython": ["*.pxd"],
        "kalmantv/eigen": ["*.pxd", "*.h"],
        "kalmantv/include/eigen" : extra_files
    },
    #package_data = packagefiles,
    # cython
    cmdclass=cmdclass,
    ext_modules=ext_modules,

    install_requires=['numpy>=1.16.4', 'scipy>=1.2.1', 'numba>=0.51.2', 'Cython>=0.29.12'],
    extras_require={'graph': ['matplotlib==3.1.0']}
    #data_files = datafiles,
    )