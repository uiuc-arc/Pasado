import time
import gc
import sys
import random

sys.path.insert(1, '../forward_mode_non_tensorized_src')
from tqdm import trange

from runge_kutta import *
from dual_intervals import *


# https://www2.oberlin.edu/math/faculty/walsh/UMAPclimate.pdf
# https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php


def climate_scalar(y0_l, y0_u, R_l, R_u, Q_l, Q_u, alpha_l, alpha_u, sigma_l, sigma_u, h, n, num_samples=100):
    merged_ys = [[] for _ in range(n + 1)]

    for _ in range(num_samples):
        y0_real = random.uniform(y0_l, y0_u)
        y0_dual = 1
        y0 = Dual(y0_real, y0_dual)

        R_real = random.uniform(R_l, R_u)
        R_dual = 0
        R = Dual(R_real, R_dual)

        Q_real = random.uniform(Q_l, Q_u)
        Q_dual = 0
        Q = Dual(Q_real, Q_dual)

        alpha_real = random.uniform(alpha_l, alpha_u)
        alpha_dual = 0
        alpha = Dual(alpha_real, alpha_dual)

        sigma_real = random.uniform(sigma_l, sigma_u)
        sigma_dual = 0
        sigma = Dual(sigma_real, sigma_dual)

        t0_real = 0
        t0_dual = 0
        t0 = Dual(t0_real, t0_dual)

        func = lambda t, x: ((Q * (1. + -alpha)) - (sigma * FourthDual(x))) / R

        ys = [y0]
        ts = [t0]
        for _ in range(n):
            y_new, t_new = euler(ys[-1], ts[-1], func, h)
            ys.append(y_new)
            ts.append(t_new)

        for i in range(n + 1):
            merged_ys[i].append(ys[i].dual)

    return merged_ys


def climate_interval(y0_l, y0_u, R_l, R_u, Q_l, Q_u, alpha_l, alpha_u, sigma_l, sigma_u, h, n):
    y0_real = (Interval(y0_l, y0_u))
    y0_dual = (Interval(1, 1))
    y0 = Dual(y0_real, y0_dual)

    R_real = (Interval(R_l, R_u))
    R_dual = (Interval(0, 0))
    R = Dual(R_real, R_dual)

    Q_real = (Interval(Q_l, Q_u))
    Q_dual = (Interval(0, 0))
    Q = Dual(Q_real, Q_dual)

    alpha_real = (Interval(alpha_l, alpha_u))
    alpha_dual = (Interval(0, 0))
    alpha = Dual(alpha_real, alpha_dual)
    # alpha = alpha_l

    sigma_real = (Interval(sigma_l, sigma_u))
    sigma_dual = (Interval(0, 0))
    sigma = Dual(sigma_real, sigma_dual)

    t0_real = (Interval(0, 0))
    t0_dual = (Interval(0, 0))
    t0 = Dual(t0_real, t0_dual)

    func = lambda t, x: ((Q * (1. + -alpha)) - (sigma * FourthDual(x))) / R

    ys = [y0]
    ts = [t0]
    for _ in range(n):
        y_new, t_new = euler(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return ys


# https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php
def climate_regular_zono(y0_l, y0_u, R_l, R_u, Q_l, Q_u, alpha_l, alpha_u, sigma_l, sigma_u, h, n):
    y0_real = Affine(Interval(y0_l, y0_u))
    y0_dual = Affine(Interval(1, 1))
    y0 = Dual(y0_real, y0_dual)

    R_real = Affine(Interval(R_l, R_u))
    R_dual = Affine(Interval(0, 0))
    R = Dual(R_real, R_dual)

    Q_real = Affine(Interval(Q_l, Q_u))
    Q_dual = Affine(Interval(0, 0))
    Q = Dual(Q_real, Q_dual)

    alpha_real = Affine(Interval(alpha_l, alpha_u))
    alpha_dual = Affine(Interval(0, 0))
    alpha = Dual(alpha_real, alpha_dual)

    sigma_real = Affine(Interval(sigma_l, sigma_u))
    sigma_dual = Affine(Interval(0, 0))
    sigma = Dual(sigma_real, sigma_dual)

    t0_real = Affine(Interval(0, 0))
    t0_dual = Affine(Interval(0, 0))
    t0 = Dual(t0_real, t0_dual)

    func = lambda t, x: ((Q * (1. + -alpha)) - (sigma * FourthDual(x))) / R

    ys = [y0]
    ts = [t0]
    for _ in range(n):
        y_new, t_new = euler(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return ys


# https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php
def climate_mixed_precise(y0_l, y0_u, R_l, R_u, Q_l, Q_u, alpha_l, alpha_u, sigma_l, sigma_u, h, n):
    y0_real = MixedAffine(Affine(Interval(y0_l, y0_u)))
    y0_dual = MixedAffine(Affine(Interval(1, 1)))
    y0 = Dual(y0_real, y0_dual)

    R_real = MixedAffine(Affine(Interval(R_l, R_u)))
    R_dual = MixedAffine(Affine(Interval(0, 0)))
    R = Dual(R_real, R_dual)

    Q_real = MixedAffine(Affine(Interval(Q_l, Q_u)))
    Q_dual = MixedAffine(Affine(Interval(0, 0)))
    Q = Dual(Q_real, Q_dual)

    alpha_real = MixedAffine(Affine(Interval(alpha_l, alpha_u)))
    alpha_dual = MixedAffine(Affine(Interval(0, 0)))
    alpha = Dual(alpha_real, alpha_dual)

    sigma_real = MixedAffine(Affine(Interval(sigma_l, sigma_u)))
    sigma_dual = MixedAffine(Affine(Interval(0, 0)))
    sigma = Dual(sigma_real, sigma_dual)

    t0_real = MixedAffine(Affine(Interval(0, 0)))
    t0_dual = MixedAffine(Affine(Interval(0, 0)))
    t0 = Dual(t0_real, t0_dual)

    func = lambda t, x: DividePrecise(
        (MultiplyPrecise(Q, (1. + -alpha)) - MultiplyPrecise(sigma, FourthDualPrecise(x))), R)

    ys = [y0]
    ts = [t0]
    for _ in range(n):
        y_new, t_new = euler(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return ys
