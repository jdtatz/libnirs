import ctypes
import scipy.special
import numba
from numba import vectorize
from numba.extending import get_cython_function_address, overload

addr = get_cython_function_address("scipy.special.cython_special", "j0")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_cython_j0 = functype(addr)
_vector_j0 = vectorize(nopython=True)(lambda v: _cython_j0(v))
overload(scipy.special.j0)(lambda v: (lambda v: _vector_j0(v)))

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_cython_erfc = functype(addr)
_vector_erfc = vectorize(["f8(f8)"], nopython=True)(lambda v: _cython_erfc(v))
overload(scipy.special.erfc)(lambda v: (lambda v: _vector_erfc(v)))

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
