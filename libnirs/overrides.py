import math
import cmath
import ctypes
import numpy as np
import scipy.special
import numba
import numba
import numba.cuda as cuda
import numba.cuda.cudaimpl
import numba.cuda.cudadecl
from numba import jit, vectorize
from numba.core import types
from numba.extending import get_cython_function_address, lower_builtin, overload

def _add_override(np_fn, math_fn, cmath_fn=None):
    cmath_fn = math_fn if cmath_fn is None else cmath_fn

    @cuda.cudaimpl.lower(np_fn, types.Number)
    def lower_override(context, builder, sig, args):
        if isinstance(sig.args[0], types.Complex):
            fn = cmath_fn
        else:
            fn = math_fn
        impl = context.get_function(fn, sig)
        return impl(builder, args)

    @cuda.cudadecl.registry.register
    class OverrideIntrinsicTemplate(numba.core.typing.templates.AbstractTemplate):
        key = np_fn
        def generic(self, args, kws):
            return args[0](args[0])

    cuda.cudadecl.registry.register_global(np_fn, types.Function(OverrideIntrinsicTemplate))


_add_override(np.exp, math.exp, cmath.exp)
_add_override(np.sqrt, math.sqrt, cmath.sqrt)
_add_override(np.sin, math.sin, cmath.sin)
_add_override(np.cos, math.cos, cmath.cos)
_add_override(np.sinh, math.sinh, cmath.sinh)
_add_override(np.cosh, math.cosh, cmath.cosh)
_add_override(np.arcsin, math.asin, cmath.asin)

_RP = (-4.79443220978201773821E9,1.95617491946556577543E12,-2.49248344360967716204E14,9.70862251047306323952E15)
_RQ = (4.99563147152651017219E2,1.73785401676374683123E5,4.84409658339962045305E7,1.11855537045356834862E10,2.11277520115489217587E12,3.10518229857422583814E14,3.18121955943204943306E16,1.71086294081043136091E18)
_PP = (7.96936729297347051624E-4,8.28352392107440799803E-2,1.23953371646414299388E0,5.44725003058768775090E0,8.74716500199817011941E0,5.30324038235394892183E0,9.99999999999999997821E-1)
_PQ = (9.24408810558863637013E-4,8.56288474354474431428E-2,1.25352743901058953537E0,5.47097740330417105182E0,8.76190883237069594232E0,5.30605288235394617618E0,1.00000000000000000218E0)
_QP = (-1.13663838898469149931E-2,-1.28252718670509318512E0,-1.95539544257735972385E1,-9.32060152123768231369E1,-1.77681167980488050595E2,-1.47077505154951170175E2,-5.14105326766599330220E1,-6.05014350600728481186E0)
_QQ = (6.43178256118178023184E1,8.56430025976980587198E2,3.88240183605401609683E3,7.24046774195652478189E3,5.93072701187316984827E3,2.06209331660327847417E3,2.42005740240291393179E2)
_DR1 = 5.783185962946784521175995758455807035071
_DR2 = 30.47126234366208639907816317502275584842
_SQ2OPI = 0.79788456080286535588
_PIO4 = .78539816339744830962


@jit(nopython=True)
def _polevl(x, coef):
    ans = cuda.fma(coef[0], x, coef[1])
    for i in range(2, len(coef)):
        ans = cuda.fma(ans, x, coef[i])
    return ans


@jit(nopython=True)
def _p1evl(x, coef):
    ans = x + coef[0]
    for i in range(1, len(coef)):
        ans = cuda.fma(ans, x, coef[i])
    return ans


def _scalar_j0(x):
    """Bessel function of the first kind of order 0. Adapted from "Cephes Mathematical Functions Library"."""
    x = abs(x)
    if x > 5:
        w = 5 / x
        q = 25 / (x*x)
        p = _polevl(q, _PP) / _polevl(q, _PQ)
        q = _polevl(q, _QP) / _p1evl(q, _QQ)
        xn = x - _PIO4
        p = p * np.cos(xn) - w * q * np.sin(xn)
        return p * _SQ2OPI / np.sqrt(x)
    elif x >= 1e-5:
        z = x*x
        return (z - _DR1) * (z - _DR2) * _polevl(z, _RP) / _p1evl(z, _RQ)
    else:
        return 1 - x*x/4

addr = get_cython_function_address("scipy.special.cython_special", "j0")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_cython_j0 = functype(addr)
_vector_j0 = vectorize(nopython=True)(lambda v: _cython_j0(v))
overload(scipy.special.j0)(lambda v: (lambda v: _vector_j0(v)))

addr = get_cython_function_address("scipy.special.cython_special", "erfcinv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_cython_erfcinv = functype(addr)
_vector_erfcinv = vectorize(nopython=True)(lambda v: _cython_erfcinv(v))
overload(scipy.special.erfcinv)(lambda v: (lambda v: _vector_erfcinv(v)))

addr = get_cython_function_address("scipy.special.cython_special", "gammainccinv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
_cython_gammainccinv = functype(addr)
_vector_gammainccinv = vectorize(nopython=True)(lambda a, y: _cython_gammainccinv(a, y))
overload(scipy.special.gammainccinv)(lambda a, y: (lambda a, y: _vector_gammainccinv(a, y)))


@cuda.cudaimpl.lower(scipy.special.j0, types.Number)
def _lower_cuda_j0(context, builder, sig, args):
    res = context.compile_internal(builder, _scalar_j0, sig, args)
    return numba.core.imputils.impl_ret_untracked(context, builder, sig, res)


@cuda.cudadecl.registry.register
class _J0IntrinsicTemplate(numba.core.typing.templates.AbstractTemplate):
    key = scipy.special.j0

    def generic(self, args, kws):
        return args[0](args[0])

cuda.cudadecl.registry.register_global(scipy.special.j0, types.Function(_J0IntrinsicTemplate))
