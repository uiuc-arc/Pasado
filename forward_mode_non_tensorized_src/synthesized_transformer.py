import mpmath
import numpy as np
# from mpmath import mp, fdiv, fadd, fsub, fsum, fneg, fmul, fabs, sqrt, exp, log, sin
import torch
from affapy.aa import Affine
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import math
import invert_gaussian_deriv as ig
import CubicEquationSolver


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh_deriv(x):
    return 1. - (np.tanh(x) * np.tanh(x))


def sigmoid_deriv(x):
    return (sigmoid(x) * (1.0 - sigmoid(x)))


def TanhDeepZ(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([np.tanh(lx), np.tanh(lx)])

    lambda_opt = min(tanh_deriv(lx), tanh_deriv(ux))

    mu1 = 0.5 * (np.tanh(ux) + np.tanh(lx) - (lambda_opt * (ux + lx)))

    mu2 = 0.5 * (np.tanh(ux) - np.tanh(lx) - (lambda_opt * (ux - lx)))

    alpha = mpmath.mpmathify(lambda_opt)
    dzeta = mpmath.mpmathify(mu1)
    delta = mpmath.mpmathify(mu2)

    return self._affineConstructor(alpha, dzeta, delta)


def SigmoidDeepZ(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([sigmoid(lx), sigmoid(lx)])

    lambda_opt = min(sigmoid_deriv(lx), sigmoid_deriv(ux))

    mu1 = 0.5 * (sigmoid(ux) + sigmoid(lx) - (lambda_opt * (ux + lx)))

    mu2 = 0.5 * (sigmoid(ux) - sigmoid(lx) - (lambda_opt * (ux - lx)))

    alpha = mpmath.mpmathify(lambda_opt)
    dzeta = mpmath.mpmathify(mu1)
    delta = mpmath.mpmathify(mu2)

    return self._affineConstructor(alpha, dzeta, delta)


def ExpChebyshev(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([np.exp(lx), np.exp(lx)])

    a = lx
    b = ux

    fa = np.exp(a)
    fb = np.exp(b)

    alpha = (fb - fa) / (b - a)

    u = np.log(alpha)

    r = lambda x: alpha * x + ((-alpha * b) + (fb))

    dzeta = ((np.exp(u) + r(u)) / 2.0) - (alpha * u)

    delta = abs(np.exp(u) - r(u)) / 2.0

    alpha = mpmath.mpmathify(alpha)
    dzeta = mpmath.mpmathify(dzeta)
    delta = mpmath.mpmathify(delta)

    return self._affineConstructor(alpha, dzeta, delta)


def SqrtChebyshev(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([np.sqrt(lx), np.sqrt(lx)])

    a = lx
    b = ux

    fa = np.sqrt(a)
    fb = np.sqrt(b)

    alpha = (fb - fa) / (b - a)

    u = 1. / ((2 * alpha) * (2 * alpha))

    r = lambda x: alpha * x + ((-alpha * b) + (fb))

    dzeta = ((np.sqrt(u) + r(u)) / 2.0) - (alpha * u)

    delta = abs(np.sqrt(u) - r(u)) / 2.0

    alpha = mpmath.mpmathify(alpha)
    dzeta = mpmath.mpmathify(dzeta)
    delta = mpmath.mpmathify(delta)

    return self._affineConstructor(alpha, dzeta, delta)


def LogChebyshev(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([np.log(lx), np.log(lx)])

    a = lx
    b = ux

    fa = np.log(a)
    fb = np.log(b)

    alpha = (fb - fa) / (b - a)

    u = 1. / alpha  # 1./((2*alpha)*(2*alpha))

    r = lambda x: alpha * x + ((-alpha * b) + (fb))

    dzeta = ((np.log(u) + r(u)) / 2.0) - (alpha * u)

    delta = abs(np.log(u) - r(u)) / 2.0

    alpha = mpmath.mpmathify(alpha)
    dzeta = mpmath.mpmathify(dzeta)
    delta = mpmath.mpmathify(delta)

    return self._affineConstructor(alpha, dzeta, delta)


def StandardNormalCDFDeepZ(self):
    lx, ux = float(self.interval.inf), float(self.interval.sup)

    if lx == ux:
        return Affine([norm.cdf(ux), norm.cdf(ux)])

    lambda_opt = min(norm.pdf(lx), norm.pdf(ux))

    mu1 = 0.5 * (norm.cdf(ux) + norm.cdf(lx) - (lambda_opt * (ux + lx)))

    mu2 = 0.5 * (norm.cdf(ux) - norm.cdf(lx) - (lambda_opt * (ux - lx)))

    alpha = mpmath.mpmathify(lambda_opt)
    dzeta = mpmath.mpmathify(mu1)
    delta = mpmath.mpmathify(mu2)

    return self._affineConstructor(alpha, dzeta, delta)


def AddNewNoiseSymbol(self, d):
    return self._affineConstructor(1., 0., d)


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def SigmoidPrimeProduct(x, y):
    return (sigmoid_deriv(x) * y)


# works
def inv_sigmoid(y):
    return torch.log(y / (1. - y))


def inv_tanh(y):
    return 0.5 * (torch.log(1 + y) - torch.log(1 - y))


def cuberoot(x):
    return (x ** (0.33333))


# tensorized/vectorized version
def inverse_poly_tensor(y):
    if type(y) in [float, int, np.float64, np.float32]:
        y = torch.tensor(y)
    Q = torch.tensor(-1 / 12.)
    R = 0.25 * y

    R = R.type(torch.complex64)

    J = torch.tensor(1.j)

    sqrt_arg = ((Q ** 3) + (R ** 2))

    sqrt_arg = sqrt_arg.type(torch.complex64)

    S = cuberoot(R + torch.sqrt(sqrt_arg))
    T = cuberoot(R - torch.sqrt(sqrt_arg))

    x1 = S + T + 0.5

    sqrt_3_by_2 = torch.sqrt(torch.tensor(3.)) / 2.
    x2 = -0.5 * (S + T) + 0.5 + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (sqrt_3_by_2 * J) * (S - T)

    return (x1, x2, x3)


def inverse_poly_tensor_tanh(y):
    if type(y) in [float, int, np.float64, np.float32]:
        y = torch.tensor(y)
    Q = torch.tensor(-1 / 3.)
    R = 0.25 * y 

    R = R.type(torch.complex64)

    J = torch.tensor(1.j)

    sqrt_arg = ((Q ** 3) + (R ** 2))

    sqrt_arg = sqrt_arg.type(torch.complex64)

    S = cuberoot(R + torch.sqrt(sqrt_arg))
    T = cuberoot(R - torch.sqrt(sqrt_arg))

    x1 = S + T #+ 0.5

    sqrt_3_by_2 = torch.sqrt(torch.tensor(3.)) / 2.
    x2 = -0.5 * (S + T)  + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T)  - (sqrt_3_by_2 * J) * (S - T)

    return (x1, x2, x3)


# def inverse_poly_tensor_tanh(y):
#	roots = CubicEquationSolver.solve(1,0,-1,0.5*y)
#	roots = [torch.Tensor([x]) for x in roots]
#	return roots


def bool_to_nan(x):
    b1 = torch.isreal(x)
    b1 = b1.float()
    b1[b1 < 0.5] = float('nan')
    return b1


def inverse_sigmoid_2nd_deriv(y):
    (x1, x2, x3) = inverse_poly_tensor(y)

    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    b3 = bool_to_nan(x3)  # sets any complex root to just NaN
    x3 = x3.real * b3

    inverted_x1 = (inv_sigmoid(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_sigmoid(x2))
    inverted_x3 = (inv_sigmoid(x3))

    return (inverted_x1, inverted_x2, inverted_x3)


def inverse_tanh_2nd_deriv(y):
    (x1, x2, x3) = inverse_poly_tensor_tanh(y)
    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    b3 = bool_to_nan(x3)  # sets any complex root to just NaN
    x3 = x3.real * b3

    inverted_x1 = (inv_tanh(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_tanh(x2))
    inverted_x3 = (inv_tanh(x3))

    return (inverted_x1, inverted_x2, inverted_x3)


# solves for x* such that sigmoid''(x*)C1-C2=0 on the interval [lx,ux]
# C1 will be either ly or uy
def InverseSigmoidDoublePrime(C1, C2, lx, ux):
    if C2==0:
        return [0.] 
    y = C2 / C1
    pt1, pt2, pt3 = inverse_sigmoid_2nd_deriv(y)
    pt1 = float(pt1)
    pt2 = float(pt2)
    pt3 = float(pt3)
    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)

    if not math.isnan(pt2):
        if (pt2 >= lx) and (pt2 <= ux):
            pts.append(pt2)

    if not math.isnan(pt3):
        if (pt3 >= lx) and (pt3 <= ux):
            pts.append(pt3)

    return pts


def InverseTanhDoublePrime(C1, C2, lx, ux):
    if C2==0:
        return [0.] 
    y = C2 / C1
    pt1, pt2, pt3 = inverse_tanh_2nd_deriv(y)
    pt1 = float(pt1)
    pt2 = float(pt2)
    pt3 = float(pt3)
    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)

    if not math.isnan(pt2):
        if (pt2 >= lx) and (pt2 <= ux):
            pts.append(pt2)

    if not math.isnan(pt3):
        if (pt3 >= lx) and (pt3 <= ux):
            pts.append(pt3)

    return pts


def InverseExpDoublePrime(C1, C2, lx, ux):
    y = float(C2 / C1)

    if y <= 0:
        return []

    pt1 = np.log(y)

    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)
    return pts


# Abstract transformer for the function f(x,y) = sigmoid'(x)*y
def SynthesizedSigmoidTransformer1Way(x1, y1, N=8):
    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [SigmoidPrimeProduct(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:

    sig = 0.00001
    A1 = A1 + np.random.normal(0, sig)
    assert (A1 != 0.)

    # bound the error of the linear approximation
    pass
    pts_ly = InverseSigmoidDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseSigmoidDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(SigmoidPrimeProduct(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val
#    print(A1)
#    print(C1-maxval,C1+maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res
    """
    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = (-SigmoidPrimeProduct(x_pt, y_pt) + (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val


    other_maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val_ = (SigmoidPrimeProduct(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val_ > other_maxval:
                other_maxval = val_

    max_violation_above = other_maxval
    max_violation_below = maxval
    assert(max_violation_above>0)
    assert(max_violation_below>0)
    max_val = ((max_violation_above+max_violation_below)*0.5)
    C1 = (C1 - max_violation_below) + max_val
    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, max_val)
    return res
    """


def TanhPrimeProduct(x, y):
    return tanh_deriv(x) * y


def SynthesizedTanhTransformer1Way(x1, y1, N=8):
    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [TanhPrimeProduct(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:

    sig = 0.00001
    A1 = A1 + np.random.normal(0, sig)
    assert (A1 != 0.)

    # bound the error of the linear approximation
    pass
    pts_ly = InverseTanhDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseTanhDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(TanhPrimeProduct(x_pt, y_pt) - (
                        A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res
    """
    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = (-TanhPrimeProduct(x_pt, y_pt) + (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val


    other_maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val_ = (TanhPrimeProduct(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val_ > other_maxval:
                other_maxval = val_

    max_violation_above = other_maxval
    max_violation_below = maxval
    assert(max_violation_above>0)
    assert(max_violation_below>0)
    max_val = ((max_violation_above+max_violation_below)*0.5)
    C1 = (C1 - max_violation_below) + max_val
    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, max_val)
    return res
    """


def SynthesizedExpPrimeProductTransformer(x1, y1, N=50):
    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [np.exp(a[0]) * a[1] for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:
    # assert(A1 != 0.) #NOT NEEDED

    pts_ly = InverseExpDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseExpDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs((np.exp(x_pt) * y_pt) - (
                        A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res


def InverseLogDoublePrime(C1, C2, lx, ux):
    y = float(C2 / C1)
    if y > 0:
        return []
    pt1 = np.sqrt(-1 / y) if y != 0 else np.nan

    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)
    return pts


# synthesize a transformer for 1/x*y
def SynthesizedLogPrimeProductTransformer(x1, y1, N=50):
    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [(a[1] / a[0]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    if A1 == 0.:
        A1 += np.random.normal(0, 0.00000001)

    # check the conditions:
    assert (A1 != 0.)

    pts_ly = InverseLogDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseLogDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs((y_pt / x_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res


def InverseSqrtDoublePrime(C1, C2, lx, ux):
    y = float(C2 / C1)
    if y >= 0:
        return []
    pt1 = ((-1. / (4 * y)) ** (2. / 3.))

    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)
    return pts


def SynthesizedSqrtPrimeProductTransformer(x1, y1, N=50):
    sqrtPrimeProd = lambda x, y: (0.5 / np.sqrt(x)) * y

    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [sqrtPrimeProd(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:
    assert (A1 != 0.)

    pts_ly = InverseSqrtDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseSqrtDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(sqrtPrimeProd(x_pt, y_pt) - (
                        A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res


# https://stackoverflow.com/questions/782418/python-inverse-trigonometry-particularly-arcsin
def InverseSinDoublePrime(C1, C2, lx, ux):
    y = float(C2 / C1)
    if abs(y) > 1:
        return []

    b = np.arcsin(y)
    b_u = b
    b_l = b
    pi_minus_b = np.pi - b
    pi_minus_b_u = pi_minus_b
    pi_minus_b_l = pi_minus_b

    periodic_bs = []
    periodic_pi_minus_bs = []

    while b_u < ux:
        if b_u >= lx:
            periodic_bs.append(b_u)
            b_u = b_u + (2 * np.pi)
        else:
            b_u = b_u + (2 * np.pi)

    while b_l > lx:
        if b_l <= ux:
            periodic_bs.append(b_l)
            b_l = b_l - (2 * np.pi)
        else:
            b_l = b_l - (2 * np.pi)

    while pi_minus_b_u < ux:
        if pi_minus_b_u >= lx:
            periodic_pi_minus_bs.append(pi_minus_b_u)
            pi_minus_b_u = pi_minus_b_u + (2 * np.pi)
        else:
            pi_minus_b_u = pi_minus_b_u + (2 * np.pi)

    while pi_minus_b_l > lx:
        if pi_minus_b_l <= ux:
            periodic_pi_minus_bs.append(pi_minus_b_l)
            pi_minus_b_l = pi_minus_b_l - (2 * np.pi)
        else:
            pi_minus_b_l = pi_minus_b_l - (2 * np.pi)

    pts = list(set(periodic_bs + periodic_pi_minus_bs))
    dist = lambda x, y: abs(x - y)

    tol = 1e-5
    check_distances = [dist(np.sin(inv_val), y) < tol for inv_val in
                       pts]  # applies sin to the inv/preimage point to see if it agrees with original y

    assert (all(check_distances))
    return pts


def SynthesizedSinPrimeProductTransformer(x1, y1, N=50):
    sinPrimeProd = lambda x, y: (np.cos(x)) * y

    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [sinPrimeProd(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:
    assert (A1 != 0.)

    pts_ly = InverseSinDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseSinDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(sinPrimeProd(x_pt, y_pt) - (
                        A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res


# synthesizes a transformer for x*y
def SynthesizedXYProductTransformer(x1, y1, N=50):
    XYProd = lambda x, y: (x) * y

    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    if ((ux == 0) and (lx == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [XYProd(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:
    #	assert(A1 != 0.)

    #	pts_ly = InverseSinDoublePrime(ly,A1,lx,ux)
    #	pts_uy = InverseSinDoublePrime(uy,A1,lx,ux)

    x_pts = [lx, ux]
    #	x_pts = pts_ly + pts_uy + [lx,ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(
                XYProd(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res


def NormalCDFPrimeProduct(x, y):
    return np.exp(-0.5 * x * x) * (1. / np.sqrt(2.0 * np.pi)) * y


def inverse_erf_2nd_deriv(y):
    return ig.invert_erf_2nd_deriv(y)


def InverseNormalCDFDoublePrime(C1, C2, lx, ux):
    y = C2 / C1
    pt1, pt2 = inverse_erf_2nd_deriv(y)
    pt1 = float(pt1)
    pt2 = float(pt2)

    pts = []
    if not math.isnan(pt1):
        if (pt1 >= lx) and (pt1 <= ux):
            pts.append(pt1)

    if not math.isnan(pt2):
        if (pt2 >= lx) and (pt2 <= ux):
            pts.append(pt2)

    return pts


# synthesize a transformer for Phi'(x)*y where Phi(x) is the normal CDF
def SynthesizedNormalCDFPrimeProductTransformer(x1, y1, N=50):
    # get grid points
    lx, ux = float(x1.interval.inf), float(x1.interval.sup)

    xs = np.linspace(lx, ux, N)

    ly, uy = float(y1.interval.inf), float(y1.interval.sup)

    if ((uy == 0) and (ly == 0)):
        return Affine([0, 0])

    ys = np.linspace(ly, uy, N)

    xys = np.asarray(cartesian_product(xs, ys))

    # evaluate the true function (what we are overapproximating) on the grid points
    zs = [NormalCDFPrimeProduct(a[0], a[1]) for a in xys]
    zs = np.asarray(zs)
    # print(xys)
    # print(zs)

    # perform linear regression on the grid points to get the coefficients
    reg = LinearRegression().fit(xys, zs)
    A1, B1 = reg.coef_
    C1 = reg.intercept_

    # check the conditions:
    assert (A1 != 0.)

    pts_ly = InverseNormalCDFDoublePrime(ly, A1, lx, ux)
    pts_uy = InverseNormalCDFDoublePrime(uy, A1, lx, ux)

    x_pts = pts_ly + pts_uy + [lx, ux]
    y_pts = [ly, uy]

    maxval = -np.inf
    for x_pt in x_pts:
        for y_pt in y_pts:
            val = abs(NormalCDFPrimeProduct(x_pt, y_pt) - (
                        A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
            if val > maxval:
                maxval = val

    # print(maxval)
    res = (A1 * x1) + (B1 * y1) + C1
    res = AddNewNoiseSymbol(res, maxval)
    return res

#solves for x such that 6x*C1-C2=0 on the interval [lx,ux]
#C1 will be either ly or uy
def InverseCubeDoublePrime(C1,C2,lx,ux):
	y = C2/(C1*6)

	pt1 = float(y)

	pts = []

	if (pt1 >= lx) and (pt1 <= ux):
		pts.append(pt1)

	return pts



#Abstract transformer for the function f(x,y) = 3x^2*y
def SynthesizedCubePrimeProductTransformer(x1,y1,N=8):
	CubePrimeProduct = lambda x,y : (3*x*x*y)

	#get grid points
	lx, ux = float(x1.interval.inf), float(x1.interval.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.interval.inf), float(y1.interval.sup)

	if ((uy==0) and (ly==0)):
		return Affine([0, 0])

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [CubePrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:

	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseCubeDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseCubeDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(CubePrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1)+(B1*y1)+C1
	res = AddNewNoiseSymbol(res,maxval)
	return res



#solves for x* such that 12x^2*C1-C2=0 on the interval [lx,ux]
#C1 will be either ly or uy
def InverseFourthDoublePrime(C1,C2,lx,ux):
	y = abs(np.sqrt(C2/(C1*12)))

	pt1 = float(y)
	pt2 = -pt1

	pts = []

	if (pt1 >= lx) and (pt1 <= ux):
		pts.append(pt1)

	if (pt2 >= lx) and (pt2 <= ux):
		pts.append(pt2)

	return pts


#Abstract transformer for the function f(x,y) = 3x^2*y
def SynthesizedFourthPrimeProductTransformer(x1,y1,N=8):
	FourthPrimeProduct = lambda x,y : (4*x*x*x*y)

	#get grid points
	lx, ux = float(x1.interval.inf), float(x1.interval.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.interval.inf), float(y1.interval.sup)

	if ((uy==0) and (ly==0)):
		return Affine([0, 0])

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [FourthPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:

	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseFourthDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseFourthDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(FourthPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1)+(B1*y1)+C1
	res = AddNewNoiseSymbol(res,maxval)
	return res

"""
def IntervalBoundTransformer1Way(x1,y1):
	#get grid points
	lx, ux = float(x1.interval.inf), float(x1.interval.sup)

	ly, uy = float(y1.interval.inf), float(y1.interval.sup)

	y_pts = [ly,uy]
	x_pts = [lx,ux]


	#if the [lx,ux] interval contains zero, we need to check the root x=0
	if not((lx>0) or (ux<0)):
		x_pts.append(0)


	maxval =-np.inf
	minval = np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = SigmoidPrimeProduct(x_pt,y_pt) #Evaluate the whole expression f'(x)y
			if val > maxval:
				maxval = val
			if val < minval:
				minval = val

	return [minval,maxval]


def empirical_bound1Way(a,b,N=500):
	#get grid points
	lx, ux = float(a.interval.inf), float(a.interval.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(b.interval.inf), float(b.interval.sup)
	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))

	zs = [SigmoidPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	return (min(zs),max(zs))









#Abstract transformer for the function f(x,y) = sigmoid'(x)*y
def RefinedSynthesizedSigmoidTransformer1Way(x1,y1,xs,ys,N=50):

	#get grid points
	lx, ux = max(float(x1.interval.inf),xs[0]), min(float(x1.interval.sup),xs[1])

	xs = np.linspace(lx,ux,N)

	ly, uy = max(float(y1.interval.inf),ys[0]), min(float(y1.interval.sup),ys[1])
	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [SigmoidPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseSigmoidDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseSigmoidDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(SigmoidPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1)+(B1*y1)+C1
	res = AddNewNoiseSymbol(res,maxval)
	return res


def RefinedIntervalBoundTransformer1Way(x1,y1,xs,ys):
	#get grid points
	lx, ux = max(float(x1.interval.inf),xs[0]), min(float(x1.interval.sup),xs[1])

	ly, uy = max(float(y1.interval.inf),ys[0]), min(float(y1.interval.sup),ys[1])

	y_pts = [ly,uy]
	x_pts = [lx,ux]


	#if the [lx,ux] interval contains zero, we need to check the root x=0
	if not((lx>0) or (ux<0)):
		x_pts.append(0)


	maxval =-np.inf
	minval = np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = SigmoidPrimeProduct(x_pt,y_pt) #Evaluate the whole expression f'(x)y
			if val > maxval:
				maxval = val
			if val < minval:
				minval = val

	return [minval,maxval]
"""
