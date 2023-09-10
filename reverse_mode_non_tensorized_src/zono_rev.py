from affapy.aa import Affine
import sys
sys.path.insert(1, '../forward_mode_non_tensorized_src')
from dual_intervals import *
from typing import Sequence


class RevAffine:
    """
    https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    def __init__(self, val, _children=(), _op=''):
        if isinstance(val, Affine):
            self.val = val
        elif isinstance(val, (tuple, list)):
            self.val = Affine(val)
        else:
            raise ValueError("Invalid input value!")
        self.grad = Affine([0., 0.])
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return str(self.val.interval)

    def __add__(self, other):
        return z_add(self, other)

    def __mul__(self, other):
        return z_mul(self, other)

    def backward(self):
        topo = list()
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = Affine([1., 1.])
        for v in reversed(topo):
            v._backward()


def z_const(_v: float):
    return RevAffine([_v, _v])


def z_mul(_a: RevAffine, _b: RevAffine):
    res = RevAffine(_a.val * _b.val, (_a, _b), 'z_mul')

    def _backward():
        _a.grad += _b.val * res.grad
        _b.grad += _a.val * res.grad

    res._backward = _backward

    return res


def z_scale(_a: float, _b: RevAffine):
    res = RevAffine(_a * _b.val, (_b,), 'z_scale')

    def _backward():
        _b.grad += _a * res.grad

    res._backward = _backward

    return res


def z_linear_transform(_a: Sequence[float], _b: Sequence[RevAffine]):
    res = RevAffine(sum([_a[i] * _b[i].val for i in range(len(_a))]), tuple(_b), 'z_linear_transform')

    def _backward():
        for i in range(len(_b)):
            _b[i].grad += _a[i] * res.grad

    res._backward = _backward

    return res


def precise_z_mul(_a: RevAffine, _b: RevAffine):
    res = RevAffine(_a.val * _b.val, (_a, _b), 'precise_z_mul')

    def _backward():
        _a.grad += SynthesizedXYProductTransformer(_b.val, res.grad)
        _b.grad += SynthesizedXYProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res


def z_add(_a: RevAffine, _b: RevAffine):
    res = RevAffine(_a.val + _b.val, (_a, _b), 'z_add')

    def _backward():
        _a.grad += res.grad
        _b.grad += res.grad

    res._backward = _backward

    return res


def z_div(_a: RevAffine, _b: RevAffine):
    res = RevAffine(_a.val / _b.val, (_a, _b), 'z_div')

    def _backward():
        _a.grad += res.grad / _b.val
        _b.grad += -(res.val / _b.val) * res.grad

    res._backward = _backward

    return res


def z_sigmoid(_a: RevAffine):
    res = RevAffine(Sigmoid(_a.val), (_a,), 'z_sigmoid')

    def _backward():
        _a.grad += res.val * (1. - res.val) * res.grad

    res._backward = _backward

    return res


def z_tanh(_a: RevAffine):
    res = RevAffine(Tanh(_a.val), (_a,), 'z_tanh')

    def _backward():
        _a.grad += (1. - res.val * res.val) * res.grad

    res._backward = _backward

    return res


def z_sin(_a: RevAffine):
    res = RevAffine(Sin(_a.val), (_a,), 'z_sin')

    def _backward():
        _a.grad += Cos(_a.val) * res.grad

    res._backward = _backward

    return res


def z_exp(_a: RevAffine):
    res = RevAffine(Exp(_a.val), (_a,), 'z_exp')

    def _backward():
        _a.grad += res.val * res.grad

    res._backward = _backward

    return res


def z_sqrt(_a: RevAffine):
    res = RevAffine(Sqrt(_a.val), (_a,), 'z_sqrt')

    def _backward():
        _a.grad += 0.5 / res.val * res.grad

    res._backward = _backward

    return res


def z_log(_a: RevAffine):
    res = RevAffine(Log(_a.val), (_a,), 'z_log')

    def _backward():
        _a.grad += res.grad / _a.val

    res._backward = _backward

    return res


def z_normal_cdf(_a: RevAffine):
    res = RevAffine(NormalCDF(_a.val), (_a,), 'z_normal_cdf')

    def _backward():
        _a.grad += (1. / np.sqrt(2. * np.pi)) * (Exp(-0.5 * Square(_a.val))) * res.grad

    res._backward = _backward

    return res


def precise_z_div(_a: RevAffine, _b: RevAffine):
    res = RevAffine(_a.val / _b.val, (_a, _b), 'precise_z_div')

    def _backward():
        g1 = SynthesizedLogPrimeProductTransformer(_b.val, res.grad)
        x1 = Affine([0., 0.])
        g2 = SynthesizedQuotientRuleTransformer(_a.val, x1, _b.val, res.grad)
        _a.grad += g1
        _b.grad += g2

    res._backward = _backward

    return res


def precise_z_sqrt(_a: RevAffine):
    res = RevAffine(Sqrt(_a.val), (_a,), 'precise_z_sqrt')

    def _backward():
        _a.grad += SynthesizedSqrtPrimeProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_sigmoid(_a: RevAffine):
    res = RevAffine(Sigmoid(_a.val), (_a,), 'precise_z_sigmoid')

    def _backward():
        _a.grad += SynthesizedSigmoidTransformer1Way(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_exp(_a: RevAffine):
    res = RevAffine(Exp(_a.val), (_a,), 'precise_z_exp')

    def _backward():
        _a.grad += SynthesizedExpPrimeProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_tanh(_a: RevAffine):
    res = RevAffine(Tanh(_a.val), (_a,), 'precise_z_tanh')

    def _backward():
        _a.grad += SynthesizedTanhTransformer1Way(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_log(_a: RevAffine):
    res = RevAffine(Log(_a.val), (_a,), 'precise_z_log')

    def _backward():
        _a.grad += SynthesizedLogPrimeProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_sin(_a: RevAffine):
    res = RevAffine(Sin(_a.val), (_a,), 'precise_z_sin')

    def _backward():
        _a.grad += SynthesizedSinPrimeProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_z_normal_cdf(_a: RevAffine):
    res = RevAffine(NormalCDF(_a.val), (_a,), 'precise_z_normal_cdf')

    def _backward():
        _a.grad += SynthesizedNormalCDFPrimeProductTransformer(_a.val, res.grad)

    res._backward = _backward

    return res
