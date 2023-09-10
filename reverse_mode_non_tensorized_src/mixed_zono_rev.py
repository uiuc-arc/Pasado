from affapy.aa import Affine
import sys
sys.path.insert(1, '../forward_mode_non_tensorized_src/')
from dual_intervals import *
from MixedAAIA import MixedAffine
from typing import Sequence


class RevMixedAffine:
    """
    https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    def __init__(self, val, _children=(), _op=''):
        if isinstance(val, MixedAffine):
            self.val = val
        elif isinstance(val, Affine):
            self.val = MixedAffine(val)
        elif isinstance(val, (tuple, list)):
            self.val = MixedAffine(Affine(val))
        else:
            raise ValueError("Invalid input value!")
        self.grad = MixedAffine(Affine([0., 0.]))
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return str(self.val.affine_form.interval)

    def __add__(self, other):
        return mixed_z_add(self, other)

    def __mul__(self, other):
        return mixed_z_mul(self, other)

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

        self.grad = MixedAffine(Affine([1., 1.]))
        for v in reversed(topo):
            v._backward()


def mixed_z_const(_v: float):
    return RevMixedAffine([_v, _v])


def mixed_z_scale(_a: float, _b: RevMixedAffine):
    res = RevMixedAffine(_a * _b.val, (_b,), 'mixed_z_scale')

    def _backward():
        _b.grad += _a * res.grad

    res._backward = _backward

    return res


def mixed_z_linear_transform(_a: Sequence[float], _b: Sequence[RevMixedAffine]):
    res = RevMixedAffine(sum([_a[i] * _b[i].val for i in range(len(_a))]), tuple(_b), 'mixed_z_linear_transform')

    def _backward():
        for i in range(len(_b)):
            _b[i].grad += _a[i] * res.grad

    res._backward = _backward

    return res


def mixed_z_mul(_a: RevMixedAffine, _b: RevMixedAffine):
    res = RevMixedAffine(_a.val * _b.val, (_a, _b), 'mixed_z_mul')

    def _backward():
        _a.grad += _b.val * res.grad
        _b.grad += _a.val * res.grad

    res._backward = _backward

    return res


def precise_mixed_z_mul(_a: RevMixedAffine, _b: RevMixedAffine):
    res = RevMixedAffine(_a.val * _b.val, (_a, _b), 'precise_mixed_z_mul')

    def _backward():
        _a.grad += SynthesizedXYProductTransformer_mixed(_b.val, res.grad)
        _b.grad += SynthesizedXYProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def mixed_z_add(_a: RevMixedAffine, _b: RevMixedAffine):
    res = RevMixedAffine(_a.val + _b.val, (_a, _b), 'mixed_z_add')

    def _backward():
        _a.grad += res.grad
        _b.grad += res.grad

    res._backward = _backward

    return res


def mixed_z_div(_a: RevMixedAffine, _b: RevMixedAffine):
    res = RevMixedAffine(_a.val / _b.val, (_a, _b), 'mixed_z_div')

    def _backward():
        _a.grad += res.grad / _b.val
        _b.grad += -(res.val / _b.val) * res.grad

    res._backward = _backward

    return res


def mixed_z_sigmoid(_a: RevMixedAffine):
    res = RevMixedAffine(Sigmoid(_a.val), (_a,), 'mixed_z_sigmoid')

    def _backward():
        _a.grad += res.val * (1. - res.val) * res.grad

    res._backward = _backward

    return res


def mixed_z_tanh(_a: RevMixedAffine):
    res = RevMixedAffine(Tanh(_a.val), (_a,), 'mixed_z_tanh')

    def _backward():
        _a.grad += (1. - res.val * res.val) * res.grad

    res._backward = _backward

    return res


def mixed_z_exp(_a: RevMixedAffine):
    res = RevMixedAffine(Exp(_a.val), (_a,), 'mixed_z_exp')

    def _backward():
        _a.grad += res.val * res.grad

    res._backward = _backward

    return res


def mixed_z_sqrt(_a: RevMixedAffine):
    res = RevMixedAffine(Sqrt(_a.val), (_a,), 'mixed_z_sqrt')

    def _backward():
        _a.grad += 0.5 / res.val * res.grad

    res._backward = _backward

    return res


def mixed_z_log(_a: RevMixedAffine):
    res = RevMixedAffine(Log(_a.val), (_a,), 'mixed_z_log')

    def _backward():
        _a.grad += res.grad / _a.val

    res._backward = _backward

    return res


def mixed_z_normal_cdf(_a: RevMixedAffine):
    res = RevMixedAffine(NormalCDF(_a.val), (_a,), 'mixed_z_normal_cdf')

    def _backward():
        _a.grad += (1. / np.sqrt(2. * np.pi)) * (Exp(-0.5 * Square(_a.val))) * res.grad

    res._backward = _backward

    return res


def precise_mixed_z_div(_a: RevMixedAffine, _b: RevMixedAffine):
    res = RevMixedAffine(_a.val / _b.val, (_a, _b), 'precise_mixed_z_div')

    def _backward():
        g1 = SynthesizedLogPrimeProductTransformer_mixed(_b.val, res.grad)
        x1 = MixedAffine(Affine([0., 0.]))
        g2 = SynthesizedQuotientRuleTransformer_mixed(_a.val, x1, _b.val, res.grad)
        _a.grad += g1
        _b.grad += g2

    res._backward = _backward

    return res


def precise_mixed_z_sqrt(_a: RevMixedAffine):
    res = RevMixedAffine(Sqrt(_a.val), (_a,), 'precise_mixed_z_sqrt')

    def _backward():
        _a.grad += SynthesizedSqrtPrimeProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_sigmoid(_a: RevMixedAffine):
    res = RevMixedAffine(Sigmoid(_a.val), (_a,), 'precise_mixed_z_sigmoid')

    def _backward():
        _a.grad += SynthesizedSigmoidTransformer1Way_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_exp(_a: RevMixedAffine):
    res = RevMixedAffine(Exp(_a.val), (_a,), 'precise_mixed_z_exp')

    def _backward():
        _a.grad += SynthesizedExpPrimeProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_tanh(_a: RevMixedAffine):
    res = RevMixedAffine(Tanh(_a.val), (_a,), 'precise_mixed_z_tanh')

    def _backward():
        _a.grad += SynthesizedTanhTransformer1Way_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_log(_a: RevMixedAffine):
    res = RevMixedAffine(Log(_a.val), (_a,), 'precise_mixed_z_log')

    def _backward():
        _a.grad += SynthesizedLogPrimeProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_sin(_a: RevMixedAffine):
    res = RevMixedAffine(Sin(_a.val), (_a,), 'precise_mixed_z_sin')

    def _backward():
        _a.grad += SynthesizedSinPrimeProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res


def precise_mixed_z_normal_cdf(_a: RevMixedAffine):
    res = RevMixedAffine(NormalCDF(_a.val), (_a,), 'precise_mixed_z_normal_cdf')

    def _backward():
        _a.grad += SynthesizedNormalCDFPrimeProductTransformer_mixed(_a.val, res.grad)

    res._backward = _backward

    return res
