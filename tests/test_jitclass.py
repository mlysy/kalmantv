from numba import njit
from numba.extending import register_jitable
from numba.experimental import jitclass
from numba import types, typeof
import unittest

# passing callables to jitclass
# current approach is to create njited class definition inside a function
# which returns object instance.
# not sure whether this will destroy performance...


@njit
def a(x):
    return x * x


@njit
def b(x):
    return x + 5.


def aa(x):
    return x*x


def bb(x):
    return x + 5.


class _foo:
    def __init__(self, fun):
        self.fun = fun


def foo(fun):
    spec = [("fun", typeof(fun))]
    jcl = jitclass(spec)
    foo_cl = jcl(_foo)
    return foo_cl(fun)


spec = [("fun", typeof(a))]
bar = jitclass(spec)(_foo)(a)

breakpoint()

bar = foo(a)
baz = foo(b)

bar.fun(5) - a(5)
baz.fun(10) - b(10)
