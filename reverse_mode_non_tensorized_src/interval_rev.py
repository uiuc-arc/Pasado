from affapy.ia import Interval
import sys

sys.path.insert(1, '../forward_mode_non_tensorized_src')
from dual_intervals import *


class RevInterval:
    """
    https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    def __init__(self, val, _children=(), _op=''):
        if isinstance(val, Interval):
            self.val = val
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            self.val = Interval(val[0], val[1])
        else:
            raise ValueError("Invalid input value!")
        self.grad = Interval(0., 0.)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return str(self.val)

    def __add__(self, other):
        return i_add(self, other)

    def __mul__(self, other):
        return i_mul(self, other)

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

        self.grad = Interval(1., 1.)
        for v in reversed(topo):
            v._backward()


def i_const(_v: float):
    return RevInterval([_v, _v])


def i_mul(_a: RevInterval, _b: RevInterval):
    res = RevInterval(_a.val * _b.val, (_a, _b), 'i_mul')

    def _backward():
        _a.grad += _b.val * res.grad
        _b.grad += _a.val * res.grad

    res._backward = _backward

    return res


def i_scale(_a: float, _b: RevInterval):
    res = RevInterval(_a * _b.val, (_b,), 'i_scale')

    def _backward():
        _b.grad += _a * res.grad

    res._backward = _backward

    return res


def i_add(_a: RevInterval, _b: RevInterval):
    res = RevInterval(_a.val + _b.val, (_a, _b), 'i_add')

    def _backward():
        _a.grad += res.grad
        _b.grad += res.grad

    res._backward = _backward

    return res


def i_div(_a: RevInterval, _b: RevInterval):
    res = RevInterval(_a.val / _b.val, (_a, _b), 'i_div')

    def _backward():
        _a.grad += res.grad / _b.val
        _b.grad += -(res.val / _b.val) * res.grad

    res._backward = _backward

    return res


def i_sigmoid(_a: RevInterval):
    res = RevInterval(Sigmoid(_a.val), (_a,), 'i_sigmoid')

    def _backward():
        _a.grad += res.val * (1. - res.val) * res.grad

    res._backward = _backward

    return res


def i_tanh(_a: RevInterval):
    res = RevInterval(Tanh(_a.val), (_a,), 'i_tanh')

    def _backward():
        _a.grad += (1. - res.val * res.val) * res.grad

    res._backward = _backward

    return res


def i_sin(_a: RevInterval):
    res = RevInterval(Sin(_a.val), (_a,), 'i_sin')

    def _backward():
        _a.grad += Cos(_a.val) * res.grad

    res._backward = _backward

    return res


def i_exp(_a: RevInterval):
    res = RevInterval(Exp(_a.val), (_a,), 'i_exp')

    def _backward():
        _a.grad += res.val * res.grad

    res._backward = _backward

    return res


def i_sqrt(_a: RevInterval):
    res = RevInterval(Sqrt(_a.val), (_a,), 'i_sqrt')

    def _backward():
        _a.grad += Interval(0.5, 0.5) / res.val * res.grad

    res._backward = _backward

    return res


def i_log(_a: RevInterval):
    res = RevInterval(Log(_a.val), (_a,), 'i_log')

    def _backward():
        _a.grad += res.grad / _a.val

    res._backward = _backward

    return res


def i_normal_cdf(_a: RevInterval):
    res = RevInterval(NormalCDF(_a.val), (_a,), 'i_normal_cdf')

    def _backward():
        _a.grad += (1. / np.sqrt(2. * np.pi)) * (Exp(-0.5 * Square(_a.val))) * res.grad

    res._backward = _backward

    return res
