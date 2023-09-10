from affapy.ia import Interval
from affapy.aa import Affine
import numpy as np
from synthesized_transformer import *
from quotient_rule import *
from synthesized_transformer_mixed import *

def Square(x):
    if type(x) in [float, int, np.float64]:
        return x * x
    elif type(x) is Affine:
        return (x * x)  # Affine.sqr(x)
    elif type(x) is Interval:
        return (x ** 2)
    elif type(x) is MixedAffine:
        return (x * x) 


def Cube(x):
    if type(x) in [float, int, np.float64]:
        return x * x * x
    elif type(x) is Affine:
        return (x * x * x)  # Affine.sqr(x)
    elif type(x) is Interval:
        return (x ** 3)
    elif type(x) is MixedAffine:
        return (x * x * x) 


def Fourth(x):
    if type(x) in [float, int, np.float64]:
        return x * x * x * x
    elif type(x) is Affine:
        return Square(Square(x))#(x * x * x * x)  # Affine.sqr(x)
    elif type(x) is Interval:
        return (x ** 4)
    elif type(x) is MixedAffine:
        return Square(Square(x))#(x * x * x * x) 


def Sqrt(x):
    if type(x) in [float, int, np.float64]:
        return np.sqrt(x)
    elif type(x) is Affine:
        return SqrtChebyshev(x)
    elif type(x) is Interval:
        return Interval.sqrt(x)
    elif type(x) is MixedAffine:
        return SqrtChebyshev_mixed(x)


def Exp(x):
    if type(x) in [float, int, np.float64]:
        return np.exp(x)
    elif type(x) is Affine:
        return ExpChebyshev(x)
    elif type(x) is Interval:
        return Interval.exp(x)
    elif type(x) is MixedAffine:
        return ExpChebyshev_mixed(x)


def Log(x):
    if type(x) in [float, int, np.float64]:
        return np.log(x)
    elif type(x) is Affine:
        return LogChebyshev(x)
    elif type(x) is Interval:
        return Interval.log(x)
    elif type(x) is MixedAffine:
        return LogChebyshev_mixed(x)


def Sin(x):
    if type(x) in [float, int, np.float64]:
        return np.sin(x)
    elif type(x) is Affine:
        return Affine.sin(x)
    elif type(x) is Interval:
        return Interval.sin(x)


def Cos(x):
    if type(x) in [float, int, np.float64]:
        return np.cos(x)
    elif type(x) is Affine:
        return Affine.cos(x)
    elif type(x) is Interval:
        return Interval.cos(x)


def Tanh(x):
    if type(x) in [float, int, np.float64]:
        return np.tanh(x)
    elif type(x) is Affine:
        return TanhDeepZ(x)
    elif type(x) is MixedAffine:
        return TanhDeepZ_mixed(x)
    elif type(x) is Interval:
        l = float(x.inf)
        u = float(x.sup)
        nl = np.tanh(l)
        nu = np.tanh(u)
        return Interval(nl, nu)




def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def Sigmoid(x):
    if type(x) in [float, int, np.float64]:
        return np_sigmoid(x)
    elif type(x) is Affine:
        return SigmoidDeepZ(x)
    elif type(x) is MixedAffine:
        return SigmoidDeepZ_mixed(x)
    elif type(x) is Interval:
        l = float(x.inf)
        u = float(x.sup)
        nl = sigmoid(l)
        nu = sigmoid(u)
        return Interval(nl, nu)




def NormalPDF(x):
    if type(x) in [float, int, np.float64]:
        return norm.pdf(x)

def NormalCDF(x):
    if type(x) in [float, int, np.float64]:
        return norm.cdf(x)
    elif type(x) is Affine:
        return StandardNormalCDFDeepZ(x)
    elif type(x) is MixedAffine:
        return StandardNormalCDFDeepZ_mixed(x)
    elif type(x) is Interval:
        l = float(x.inf)
        u = float(x.sup)
        nl = norm.cdf(l)
        nu = norm.cdf(u)
        return Interval(nl, nu)




class Dual:

    def __init__(self, real_, dual_):
        self.real = real_
        self.dual = dual_

    def __add__(self, other):
        if type(other) is Dual:
            real = self.real + other.real
            dual = self.dual + other.dual
            return Dual(real, dual)
        elif type(other) in [float, int, np.float64]:
            real = self.real + other
            dual = self.dual
            return Dual(real, dual)

    __radd__ = __add__

    def __sub__(self, other):
        rhs = -1 * other
        return self + rhs

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if type(other) is Dual:
            real = self.real * other.real
            dual = (self.dual * other.real) + (self.real * other.dual)
            return Dual(real, dual)
        elif type(other) in [float, int, np.float64]:
            real = self.real * other
            dual = self.dual * other
            return Dual(real, dual)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) is Dual:
            real = self.real / other.real
            dual = ((self.dual * other.real) - (self.real * other.dual)) / (Square(other.real))
            return Dual(real, dual)
        elif type(other) in [float, int, np.float64]:
            real = self.real / other
            dual = self.dual / other
            return Dual(real, dual)

    def __neg__(self):
        real = -self.real
        dual = -self.dual
        return Dual(real, dual)


def SigmoidDual(d):
    real = Sigmoid(d.real)
    #	dual = Sigmoid(d.real)*(1-Sigmoid(d.real))*d.dual
    dual = real * (1 + -real) * d.dual
    return Dual(real, dual)


def TanhDual(d):
    real = Tanh(d.real)
    dual = (1 - Square(real)) * d.dual
    return Dual(real, dual)


def ExpDual(d):
    real = Exp(d.real)
    dual = real * d.dual
    return Dual(real, dual)


def SinDual(d):
    real = Sin(d.real)
    dual = Cos(real) * d.dual
    return Dual(real, dual)


def LogDual(d):
    real = Log(d.real)
    dual = d.dual / d.real
    return Dual(real, dual)

def SquareDual(d):
    real = Square(d.real)
    dual = d.real * d.dual * 2.
    return Dual(real, dual)

def CubeDual(d):
    real = Cube(d.real)
    dual = 3.*Square(d.real) * d.dual 
    return Dual(real, dual)

def FourthDual(d):
    real = Fourth(d.real)
    dual = 4.*Cube(d.real) * d.dual 
    return Dual(real, dual)

def SqrtDual(d):
    real = Sqrt(d.real)
    dual = (d.dual / (2.0 * real))
    return Dual(real, dual)


def NormalCDFDual(d):
    c = float(1.0 / np.sqrt(2 * np.pi))
    real = NormalCDF(d.real)
    dual = (Exp(Square(d.real) * (-0.5)) * c) * d.dual
    return Dual(real, dual)


# Precise Zonotopes
def SigmoidDualPrecise(d):
    x = d.real
    y = d.dual
    real = Sigmoid(d.real)
    if type(x) is Affine:
        dual = SynthesizedSigmoidTransformer1Way(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedSigmoidTransformer1Way_mixed(x, y)
        return Dual(real, dual)

def TanhDualPrecise(d):
    x = d.real
    y = d.dual
    real = Tanh(d.real)
    if type(x) is Affine:
        dual = SynthesizedTanhTransformer1Way(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedTanhTransformer1Way_mixed(x, y)
        return Dual(real, dual)


def ExpDualPrecise(d):
    x = d.real
    y = d.dual
    real = Exp(d.real)
    if type(x) is Affine:
        dual = SynthesizedExpPrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedExpPrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)

def SinDualPrecise(d):
    x = d.real
    y = d.dual
    real = Sin(d.real)
    dual = SynthesizedSinPrimeProductTransformer(x, y)
    return Dual(real, dual)


def LogDualPrecise(d):
    x = d.real
    y = d.dual
    real = Log(d.real)
    if type(x) is Affine:
        dual = SynthesizedLogPrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedLogPrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)


def SqrtDualPrecise(d):
    x = d.real
    y = d.dual
    real = Sqrt(d.real)
    if type(x) is Affine:
        dual = SynthesizedSqrtPrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedSqrtPrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)


def CubeDualPrecise(d):
    x = d.real
    y = d.dual
    real = Cube(d.real)
    if type(x) is Affine:
        dual = SynthesizedCubePrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedCubePrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)



def FourthDualPrecise(d):
    x = d.real
    y = d.dual
    real = Fourth(d.real)
    if type(x) is Affine:
        dual = SynthesizedFourthPrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedFourthPrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)


def DividePrecise(d1, d2):
    real = d1.real / d2.real
    if type(real) is Affine:
        dual = SynthesizedQuotientRuleTransformer(d1.real, d1.dual, d2.real, d2.dual)
        return Dual(real, dual)
    elif type(real) is MixedAffine:
        dual = SynthesizedQuotientRuleTransformer_mixed(d1.real, d1.dual, d2.real, d2.dual)
        return Dual(real, dual)


def MultiplyPrecise(d1, d2):
    if (type(d1) in [float, int, np.float64]):
        return d1*d2
    if (type(d2) in [float, int, np.float64]):
        return d1*d2

    real = d1.real * d2.real
    if type(real) is Affine:
        dual = SynthesizedProductRuleTransformer(d1.real, d1.dual, d2.real, d2.dual)
        return Dual(real, dual)
    elif type(real) is MixedAffine:
        dual = SynthesizedProductRuleTransformer_mixed(d1.real, d1.dual, d2.real, d2.dual)
        return Dual(real, dual)



def NormalCDFDualPrecise(d):
    x = d.real
    y = d.dual
    real = NormalCDF(x)
    if type(x) is Affine:
        dual = SynthesizedNormalCDFPrimeProductTransformer(x, y)
        return Dual(real, dual)
    elif type(x) is MixedAffine:
        dual = SynthesizedNormalCDFPrimeProductTransformer_mixed(x, y)
        return Dual(real, dual)

