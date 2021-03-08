from __future__ import annotations

from typing import Callable, Generic, NamedTuple, Optional, Type, TypeVar

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from numba import typeof
from numba.extending import overload_attribute

from .utils import jit, fma

try:
    from math import comb
except ImportError:
    # Python 3.7 compat
    def comb(n: int, k: int, /) -> int:
        from math import factorial
        if k <= n:
            return factorial(n) // (factorial(k) * factorial(n - k))
        else:
            return 0


def weighted_quantile(dataset: ArrayLike, weight: ArrayLike, q: Optional[ArrayLike]=None, axis: Optional[int]=None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    s_idx = np.argsort(dataset)
    x = np.take_along_axis(dataset, s_idx, axis=axis)
    w = np.take_along_axis(weight, s_idx, axis=axis)
    cdf = (np.cumsum(w, axis=axis) - (w / 2)) / np.sum(w, axis=axis)
    if q is not None:
        return np.interp(q, cdf, x)
    else:
        return cdf, x


T = TypeVar("T")
V = TypeVar("V")


class classproperty(Generic[T, V]):
    name: str
    owner: type[T]
    # fget: Callable[[type[T]], V]

    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner

    def __init__(self, fget: Callable[[type[T]], V]):
        # classmethods can't be called directly, so get inner function
        # if `fget` is a classmethod
        self.fget = getattr(fget, "__func__", fget)

    def __get__(self, instance: Optional[T], owner: Optional[type[T]]=None) -> V:
        if owner is None and instance is not None:
            owner = type(instance)
        if not hasattr(self, "owner") and owner is not None:
            self.owner = owner
        assert self.owner is not None
        assert owner is None or issubclass(owner, self.owner)
        assert instance is None or isinstance(instance, self.owner)
        fget = self.fget
        return fget(self.owner)


class StandardMoments(NamedTuple):
    mean: np.ndarray
    var: np.ndarray
    skew: np.ndarray
    kurt: np.ndarray


class CentralMoments(NamedTuple):
    '''
    Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments
    Philippe P ́ebay
    https://core.ac.uk/download/pdf/205797606.pdf

    Simpler Online Updates for Arbitrary-Order Central Moment
    Xiangrui Meng
    https://arxiv.org/pdf/1510.04923.pdf
    '''
    n: np.ndarray
    mu: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    m4: np.ndarray

    @classmethod
    def alloc(cls, shape=(), scalar_dtype: DTypeLike=np.float64) -> CentralMoments:
        return cls._make(np.zeros(shape, dtype=scalar_dtype) for _ in cls._fields)

    @classmethod
    def _dtype(cls, scalar_dtype: DTypeLike=np.float64) -> DTypeLike:
        return np.dtype([(f, scalar_dtype) for f in cls._fields], align=True)

    @classproperty
    @classmethod
    def dtype(cls) -> DTypeLike:
        return cls._dtype()

    @classmethod
    def from_array(cls, array: np.ndarray) -> CentralMoments:
        assert array.dtype == cls.dtype
        return cls(**{f: array[f] for f in cls._fields})

    def to_array(self) -> np.ndarray:
        cls = type(self)
        return np.rec.fromarrays(self, dtype=cls.dtype)

    @jit
    def push(self, x: ArrayLike, index=()):
        self.n[index] += 1
        delta = x - self.mu[index]
        d_n = delta / self.n[index]
        self.mu[index] += d_n
        s = -delta
        self.m2[index] += fma(s, d_n, delta**2)
        s = -delta
        s = fma(s, d_n, -3 * self.m2[index])
        self.m3[index] += fma(s, d_n, delta**3)
        s = -delta
        s = fma(s, d_n, -6 * self.m2[index])
        s = fma(s, d_n, -4 * self.m3[index])
        self.m4[index] += fma(s, d_n, delta**4)

    def merge(self, rhs: CentralMoments) -> CentralMoments:
        n = self.n + rhs.n
        delta = rhs.mu - self.mu
        d_n = delta / n
        mu = (self.mu * self.n + rhs.mu * rhs.n) / n
        s = d_n * n * self.n * rhs.n * (self.n**1 + rhs.n**1)
        m2 = fma(s, d_n, self.m2 + rhs.m2)
        s = d_n * n * self.n * rhs.n * (self.n**2 - rhs.n**2)
        s = fma(s, d_n, 3 * (self.n**1 * rhs.m2 - rhs.n**1 * self.m2))
        m3 = fma(s, d_n, self.m3 + rhs.m3)
        s = d_n * n * self.n * rhs.n * (self.n**3 + rhs.n**3)
        s = fma(s, d_n, 6 * (self.n**2 * rhs.m2 + rhs.n**2 * self.m2))
        s = fma(s, d_n, 4 * (self.n**1 * rhs.m3 - rhs.n**1 * self.m3))
        m4 = fma(s, d_n, self.m4 + rhs.m4)
        return CentralMoments(n, mu, m2, m3, m4)

    def update_to_n(self, n: int) -> CentralMoments:
        n_0 = n - self.n
        return self.merge(CentralMoments(n_0, *(np.zeros(()) for _ in self._fields[1:])))

    def standardize(self) -> StandardMoments:
        mean = self.mu
        var = self.m2 / self.n
        skew = np.sqrt(self.n) * self.m3 / self.m2 ** (3 / 2)
        kurt = self.n * self.m4 / self.m2 ** 2
        return StandardMoments(mean, var, skew, kurt)

    @property
    def standard(self):
        return self.standardize()


class WeightedCentralMoments(NamedTuple):
    '''
    Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments
    Philippe P ́ebay
    https://core.ac.uk/download/pdf/205797606.pdf

    Simpler Online Updates for Arbitrary-Order Central Moment
    Xiangrui Meng
    https://arxiv.org/pdf/1510.04923.pdf
    '''
    W: np.ndarray
    mu: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    m4: np.ndarray

    @classmethod
    def alloc(cls, shape=(), scalar_dtype=np.float64) -> WeightedCentralMoments:
        return cls._make(np.zeros(shape, dtype=scalar_dtype) for _ in cls._fields)

    @classmethod
    def _dtype(cls, scalar_dtype=np.float64) -> DTypeLike:
        return np.dtype([(f, scalar_dtype) for f in cls._fields], align=True)

    @classproperty
    @classmethod
    def dtype(cls) -> DTypeLike:
        return cls._dtype()

    @classmethod
    def from_array(cls, array: np.ndarray) -> WeightedCentralMoments:
        assert array.dtype == cls.dtype
        return cls(**{f: array[f] for f in cls._fields})

    def to_array(self) -> np.ndarray:
        cls = type(self)
        return np.rec.fromarrays(self, dtype=cls.dtype)

    @jit
    def push(self, x: ArrayLike, w: ArrayLike, index=()):
        self.W[index] += w
        delta = x - self.mu[index]
        d_w = w * delta / self.W[index]
        self.mu[index] += d_w
        s = -delta * w
        self.m2[index] += fma(s, d_w, w * delta**2)
        s = -delta * w
        s = fma(s, d_w, -3 * self.m2[index])
        self.m3[index] += fma(s, d_w, w * delta**3)
        s = -delta * w
        s = fma(s, d_w, -6 * self.m2[index])
        s = fma(s, d_w, -4 * self.m3[index])
        self.m4[index] += fma(s, d_w, w * delta**4)

    def merge(self, rhs: WeightedCentralMoments) -> WeightedCentralMoments:
        W = self.W + rhs.W
        delta = rhs.mu - self.mu
        d_w = delta / W
        mu = (self.mu * self.W + rhs.mu * rhs.W) / W
        s = d_w * W * self.W * rhs.W * (self.W**1 + rhs.W**1)
        m2 = fma(s, d_w, self.m2 + rhs.m2)
        s = d_w * W * self.W * rhs.W * (self.W**2 - rhs.W**2)
        s = fma(s, d_w, 3 * (self.W**1 * rhs.m2 - rhs.W**1 * self.m2))
        m3 = fma(s, d_w, self.m3 + rhs.m3)
        s = d_w * W * self.W * rhs.W * (self.W**3 + rhs.W**3)
        s = fma(s, d_w, 6 * (self.W**2 * rhs.m2 + rhs.W**2 * self.m2))
        s = fma(s, d_w, 4 * (self.W**1 * rhs.m3 - rhs.W**1 * self.m3))
        m4 = fma(s, d_w, self.m4 + rhs.m4)
        return WeightedCentralMoments(W, mu, m2, m3, m4)

    def standardize(self) -> StandardMoments:
        mean = self.mu
        var = self.m2 / self.W
        skew = np.sqrt(self.W) * self.m3 / self.m2 ** (3 / 2)
        kurt = self.W * self.m4 / self.m2 ** 2
        return StandardMoments(mean, var, skew, kurt)

    @property
    def standard(self):
        return self.standardize()



# TODO: Fix this hacky workaround
_CentralMoments_push = CentralMoments.push
_WeightedCentralMoments_push = WeightedCentralMoments.push

@overload_attribute(typeof(CentralMoments), 'push')
def _CentralMoments_push_attr(self):
    return lambda self: _CentralMoments_push

@overload_attribute(typeof(WeightedCentralMoments), 'push')
def _WeightedCentralMoments_push_attr(self):
    return lambda self: _WeightedCentralMoments_push



class NWeightedCentralMoments(NamedTuple):
    '''
    Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments
    Philippe P ́ebay
    https://core.ac.uk/download/pdf/205797606.pdf

    Simpler Online Updates for Arbitrary-Order Central Moment
    Xiangrui Meng
    https://arxiv.org/pdf/1510.04923.pdf
    '''
    W: np.ndarray
    mu: np.ndarray
    moments: tuple[np.ndarray, ...]

    @staticmethod
    def alloc(n, shape=(), scalar_dtype=np.float64):
        assert n >= 2
        return NWeightedCentralMoments(
            W=np.zeros(shape, dtype=scalar_dtype),
            mu=np.zeros(shape, dtype=scalar_dtype),
            moments=tuple(np.zeros(shape, dtype=scalar_dtype) for _ in range(2, n + 1))
        )

    @jit
    def push(self, x: ArrayLike, w: ArrayLike, index=()):
        self.W[index] += 1
        delta = x - self.mu[index]
        d_w = w * delta / self.W[index]
        self.mu[index] += d_w
        for i in range(len(self.moments)):
            p = i + 2
            # v1
            # self.moments[i][index] += -sum(comb(p, k) * d_w**k * self.moments[p-k-2][index] for k in range(1, p-1)) + delta * w * (delta**(p-1) - d_w**(p-1))
            # v2
            s = -delta * w
            for k in reversed(range(1, (p - 2) + 1)):
                s = fma(s, d_w, -comb(p, k) * self.moments[p-k-2][index])
            self.moments[i][index] += fma(s, d_w, w * delta**p)

    def merge(self, rhs: NWeightedCentralMoments) -> NWeightedCentralMoments:
        W = self.W + rhs.W
        delta = rhs.mu - self.mu
        d_w = delta / W
        mu = (self.mu * self.W + rhs.mu * rhs.W) / W
        moments = []
        assert len(self.moments) == len(rhs.moments)
        for i in reversed(range(len(self.moments))):
            p = i + 2
            s = d_w * W * self.W * rhs.W * (self.W**(p-1) - (-rhs.W)**(p-1))
            for k in reversed(range(1, (p - 2) + 1)):
                s = fma(s, d_w, comb(p, k) * (self.W**k * rhs.moments[p-k-2] + (-rhs.W)**k * self.moments[p-k-2]))
            s = fma(s, d_w, self.moments[i] + rhs.moments[i])
            moments.append(s)
        return NWeightedCentralMoments(W, mu, tuple(moments))


_template = """
class CentralMoments(NamedTuple):
    '''
    Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments
    Philippe P ́ebay
    https://core.ac.uk/download/pdf/205797606.pdf

    Simpler Online Updates for Arbitrary-Order Central Moment
    Xiangrui Meng
    https://arxiv.org/pdf/1510.04923.pdf
    '''
    n: np.ndarray
    mu: np.ndarray
{%- for p in range(2, n+1) %}
    m{{p}}: np.ndarray
{%- endfor %}

    @classmethod
    def alloc(cls, shape=(), scalar_dtype=np.float64):
        return cls._make(np.zeros(shape, dtype=scalar_dtype) for _ in cls._fields)

    @classmethod
    def _dtype(cls, scalar_dtype=np.float64):
        return np.dtype([(f, scalar_dtype) for f in cls._fields], align=True)

    @classproperty
    def dtype(cls):
        return cls._dtype()

    @classmethod
    def from_array(cls, array: np.ndarray) -> CentralMoments:
        assert array.dtype == cls.dtype
        return cls(**{f: array[f] for f in cls._fields})

    def to_array(self) -> np.ndarray:
        cls = type(self)
        return np.rec.fromarrays(self, dtype=cls.dtype)

    @jit
    def push(self, x: ArrayLike, index=()):
        self.n[index] += 1
        delta = x - self.mu[index]
        d_n = delta / self.n[index]
        self.mu[index] += d_n
{%- for p in range(2, n+1) %}
        s = -delta
{%- for k in range(p - 2, 0, -1) %}
        s = fma(s, d_n, -{{ comb(p, k) }} * self.m{{p - k}}[index])
{%- endfor %}
        self.m{{p}}[index] += fma(s, d_n, delta**{{p}})
{%- endfor %}

    def merge(self, rhs: CentralMoments) -> CentralMoments:
        n = self.n + rhs.n
        delta = rhs.mu - self.mu
        d_n = delta / n
        mu = (self.mu * self.n + rhs.mu * rhs.n) / n
{%- for p in range(2, n+1) %}
        s = d_n * n * self.n * rhs.n * (self.n**{{p-1}} {{ '+' if p % 2 == 0 else '-' }} rhs.n**{{p-1}})
{%- for k in range(p - 2, 0, -1) %}
        s = fma(s, d_n, {{comb(p, k)}} * (self.n**{{k}} * rhs.m{{p-k}} {{'+' if k % 2 == 0 else '-'}} rhs.n**{{k}} * self.m{{p-k}}))
{%- endfor %}
        m{{p}} = fma(s, d_n, self.m{{p}} + rhs.m{{p}})
{%- endfor %}
        return CentralMoments(n, mu {%- for p in range(2, n+1) -%} , m{{p}} {%- endfor %})

    def update_to_n(self, n: int) -> CentralMoments:
        n_0 = n - self.n
        return self.merge(CentralMoments(n_0, *(np.zeros(()) for _ in self._fields[1:])))

    def standardize(self) -> StandardMoments:
        mean = self.mu
        var = self.m2 / self.n
        inv_var = 1 / var
        inv_sigma = 1 / np.sqrt(var)
        inv_n = 1 / self.n
        return StandardMoments(mean, var {%- for p in range(3, n+1) -%} , self.m{{p}} * inv_n * {% if p % 2 == 0 -%} inv_var**{{p//2}} {%- else -%} inv_sigma**{{p}} {%- endif -%} {%- endfor %})

    @property
    def standard(self):
        return self.standardize()


class WeightedCentralMoments(NamedTuple):
    '''
    Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments
    Philippe P ́ebay
    https://core.ac.uk/download/pdf/205797606.pdf

    Simpler Online Updates for Arbitrary-Order Central Moment
    Xiangrui Meng
    https://arxiv.org/pdf/1510.04923.pdf
    '''
    W: np.ndarray
    mu: np.ndarray
{%- for p in range(2, n+1) %}
    m{{p}}: np.ndarray
{%- endfor %}

    @classmethod
    def alloc(cls, shape=(), scalar_dtype=np.float64):
        return cls._make(np.zeros(shape, dtype=scalar_dtype) for _ in cls._fields)

    @classmethod
    def _dtype(cls, scalar_dtype=np.float64):
        return np.dtype([(f, scalar_dtype) for f in cls._fields], align=True)

    @classproperty
    def dtype(cls):
        return cls._dtype()

    @classmethod
    def from_array(cls, array: np.ndarray) -> WeightedCentralMoments:
        assert array.dtype == cls.dtype
        return cls(**{f: array[f] for f in cls._fields})

    def to_array(self) -> np.ndarray:
        cls = type(self)
        return np.rec.fromarrays(self, dtype=cls.dtype)

    @jit
    def push(self, x: ArrayLike, w: ArrayLike, index=()):
        self.W[index] += w
        delta = x - self.mu[index]
        d_w = w * delta / self.W[index]
        self.mu[index] += d_w
{%- for p in range(2, n+1) %}
        s = -delta * w
{%- for k in range(p - 2, 0, -1) %}
        s = fma(s, d_w, -{{ comb(p, k) }} * self.m{{p - k}}[index])
{%- endfor %}
        self.m{{p}}[index] += fma(s, d_w, w * delta**{{p}})
{%- endfor %}

    def merge(self, rhs: WeightedCentralMoments) -> WeightedCentralMoments:
        W = self.W + rhs.W
        delta = rhs.mu - self.mu
        d_w = delta / W
        mu = (self.mu * self.W + rhs.mu * rhs.W) / W
{%- for p in range(2, n+1) %}
        s = d_w * W * self.W * rhs.W * (self.W**{{p-1}} {{ '+' if p % 2 == 0 else '-' }} rhs.W**{{p-1}})
{%- for k in range(p - 2, 0, -1) %}
        s = fma(s, d_w, {{comb(p, k)}} * (self.W**{{k}} * rhs.m{{p-k}} {{'+' if k % 2 == 0 else '-'}} rhs.W**{{k}} * self.m{{p-k}}))
{%- endfor %}
        m{{p}} = fma(s, d_w, self.m{{p}} + rhs.m{{p}})
{%- endfor %}
        return WeightedCentralMoments(W, mu {%- for p in range(2, n+1) -%} , m{{p}} {%- endfor %})

    def standardize(self) -> StandardMoments:
        mean = self.mu
        var = self.m2 / self.W
        inv_var = 1 / var
        inv_sigma = 1 / np.sqrt(var)
        inv_W = 1 / self.W
        return StandardMoments(mean, var {%- for p in range(3, n+1) -%} , self.m{{p}} * inv_W * {% if p % 2 == 0 -%} inv_var**{{p//2}} {%- else -%} inv_sigma**{{p}} {%- endif -%} {%- endfor %})

    @property
    def standard(self):
        return self.standardize()
"""


def _generate(n: int):
    import jinja2

    return jinja2.Template(_template).render(n=n, comb=comb)

