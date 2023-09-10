from affapy.ia import Interval
from affapy.aa import Affine
import numpy as np


def interval_intersection(a, b):
    return Interval(max(a.inf, b.inf), min(a.sup, b.sup))


class MixedAffine:

    def __init__(self, affine_form_, interval_=None):
        self.affine_form = affine_form_
        if interval_ is None:
            self.bounds = self.affine_form.interval
        else:
            # intersect the affine_form's interval with interval_
            self.bounds = interval_intersection(interval_, self.affine_form.interval)

    def get_bounds(self):
        return (self.bounds.inf, self.bounds.sup)

    def __add__(self, other):
        if type(other) is MixedAffine:
            new_affine_form = self.affine_form + other.affine_form
            new_bounds = self.bounds + other.bounds
            return MixedAffine(new_affine_form, new_bounds)

        elif type(other) in [float, int, np.float64]:
            new_affine_form = self.affine_form + other
            new_bounds = self.bounds + other
            return MixedAffine(new_affine_form, new_bounds)
        else:
            raise Exception

    __radd__ = __add__

    def __sub__(self, other):
        if type(other) is MixedAffine:
            new_affine_form = self.affine_form - other.affine_form
            new_bounds = self.bounds - other.bounds
            return MixedAffine(new_affine_form, new_bounds)

        elif type(other) in [float, int, np.float64]:
            new_affine_form = self.affine_form - other
            new_bounds = self.bounds - other
            return MixedAffine(new_affine_form, new_bounds)

        else:
            raise Exception

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if type(other) is MixedAffine:
            new_affine_form = self.affine_form * other.affine_form
            new_bounds = self.bounds * other.bounds
            return MixedAffine(new_affine_form, new_bounds)

        elif type(other) in [float, int, np.float64]:
            new_affine_form = self.affine_form * other
            new_bounds = self.bounds * other
            return MixedAffine(new_affine_form, new_bounds)
        else:
            raise Exception

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) is MixedAffine:
            new_affine_form = self.affine_form / other.affine_form
            new_bounds = self.bounds / other.bounds
            return MixedAffine(new_affine_form, new_bounds)

        elif type(other) in [float, int, np.float64]:
            new_affine_form = self.affine_form / other
            new_bounds = self.bounds / other
            return MixedAffine(new_affine_form, new_bounds)
        else:
            raise Exception

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            new_affine_form = other.affine_form / self.affine_form
            new_bounds = Interval(1., 1.) / self.bounds * other
            return MixedAffine(new_affine_form, new_bounds)
        elif isinstance(other, (float, int, np.float64)):
            new_affine_form = other * self.affine_form.inv()
            new_bounds = Interval(1., 1.) / self.bounds * other
            return MixedAffine(new_affine_form, new_bounds)
        else:
            raise Exception

    def __neg__(self):
        affine_form = self.affine_form * -1
        bounds = self.bounds * -1
        return MixedAffine(affine_form, bounds)
