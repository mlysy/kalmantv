import os
from .omp_init import _omp_init
from .kalmantv import _KalmanTV as KalmanTV


def get_include():
    r"""
    Return the directory that contains the `KalmanTV.h` header file.

    Similar to `numpy.get_include()`, except only works after kalmantv has been installed.
    """
    return os.path.dirname(__file__)
    # import kalmantv
    # if numpy.show_config is None:
    #     # running from numpy source directory
    #     d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    # else:
    #     # using installed numpy core headers
    #     import numpy.core as core
    #     d = os.path.join(os.path.dirname(core.__file__), 'include')
    # return d
