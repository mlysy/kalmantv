from setuptools import setup, find_packages, Extension
import numpy as np

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

# extension modules
cpp_modules = ['kalmantv']

# cpp modules
ext = '.pyx' if USE_CYTHON else '.cpp'
ext_modules = [Extension("kalmantv.cython.{}".format(mod),
                         ["include/{}".format(mod)+ext],
                         include_dirs=[
                             np.get_include(),
                             "include/eigen-3.3.7"],
                         language='c++')
               for mod in cpp_modules]

setup(
    name="kalmantv",
    version="0.1",
    author="Martin Lysy, Mohan Wu",
    author_email="mlysy@uwaterloo.ca",
    description="Kalman Filtering with Cython + Eigen",
    keywords="Kalman Eigen Cython",
    url="http://github.com/mlysy/kalmantv",
    packages=['kalmantv'],

    # cython
    cmdclass=cmdclass,
    ext_modules=ext_modules,

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy', 'scipy'],
    setup_requires=['setuptools>=38'],
)
