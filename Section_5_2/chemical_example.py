import random
import time
import gc
import sys
import os
import pickle

sys.path.insert(1, '../forward_mode_non_tensorized_src')
import numpy as np
import matplotlib.pyplot as plt
from dual_intervals import *
from runge_kutta import *
from sklearn.neural_network import MLPRegressor
from tqdm import trange

np.random.seed(31415)


def ground_truth_f(t, k1, k_1, Ca0, Ca):
    return (Ca * (-k1)) + (k_1 * (Ca0 + (-Ca)))


def Cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def generate_data():
    t = np.linspace(0, 2, 5)
    k1 = np.linspace(2.5, 3.5, 5)
    k_1 = np.linspace(2.5, 3.5, 5)
    Ca0 = np.linspace(0.5, 1.5, 5)
    Ca = np.linspace(0.5, 1.5, 5)

    data = Cartesian_product(*[t, k1, k_1, Ca0, Ca])
    # data = [x for x in data if x[3]<=x[4]]
    labels = [ground_truth_f(x[0], x[1], x[2], x[3], x[4]) for x in data]

    return data, labels


# data will be of the form:
# X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
# y = [0, 0, 0, 1]
def get_nn(fresh=False):
    X, y = generate_data()
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='tanh', hidden_layer_sizes=(5, 3, 2), random_state=1,
                       max_iter=4000)

    if fresh:
        if not os.path.isfile('_temp.pkl'):
            clf.fit(X, y)
            with open('_temp.pkl', 'wb') as f:
                pickle.dump(clf, f)
        else:
            with open('_temp.pkl', 'rb') as f:
                clf = pickle.load(f)
    else:
        with open('saved.pkl', 'rb') as f:
            clf = pickle.load(f)

    return clf


def get_dual_lifting(NN, isPrecise: bool):
    input_to_1st = NN.coefs_[0]

    def SD(x):
        return TanhDualPrecise(x) if isPrecise else TanhDual(x)

    def First_layer(t, k1, k_1, Ca0, Ca):
        i1 = SD(
            input_to_1st[0][0] * (t) + input_to_1st[1][0] * (k1) + input_to_1st[2][0] * (k_1) + input_to_1st[3][0] * (
                Ca0) + input_to_1st[4][0] * (Ca) + NN.intercepts_[0][0])
        i2 = SD(
            input_to_1st[0][1] * (t) + input_to_1st[1][1] * (k1) + input_to_1st[2][1] * (k_1) + input_to_1st[3][1] * (
                Ca0) + input_to_1st[4][1] * (Ca) + NN.intercepts_[0][1])
        i3 = SD(
            input_to_1st[0][2] * (t) + input_to_1st[1][2] * (k1) + input_to_1st[2][2] * (k_1) + input_to_1st[3][2] * (
                Ca0) + input_to_1st[4][2] * (Ca) + NN.intercepts_[0][2])
        i4 = SD(
            input_to_1st[0][3] * (t) + input_to_1st[1][3] * (k1) + input_to_1st[2][3] * (k_1) + input_to_1st[3][3] * (
                Ca0) + input_to_1st[4][3] * (Ca) + NN.intercepts_[0][3])
        i5 = SD(
            input_to_1st[0][4] * (t) + input_to_1st[1][4] * (k1) + input_to_1st[2][4] * (k_1) + input_to_1st[3][4] * (
                Ca0) + input_to_1st[4][4] * (Ca) + NN.intercepts_[0][4])
        return (i1, i2, i3, i4, i5)

    first_to_2nd = NN.coefs_[1]

    def Second_layer(x1, x2, x3, x4, x5):
        i1 = SD(
            first_to_2nd[0][0] * (x1) + first_to_2nd[1][0] * (x2) + first_to_2nd[2][0] * (x3) + first_to_2nd[3][0] * (
                x4) + first_to_2nd[4][0] * (x5) + NN.intercepts_[1][0])
        i2 = SD(
            first_to_2nd[0][1] * (x1) + first_to_2nd[1][1] * (x2) + first_to_2nd[2][1] * (x3) + first_to_2nd[3][1] * (
                x4) + first_to_2nd[4][1] * (x5) + NN.intercepts_[1][1])
        i3 = SD(
            first_to_2nd[0][2] * (x1) + first_to_2nd[1][2] * (x2) + first_to_2nd[2][2] * (x3) + first_to_2nd[3][2] * (
                x4) + first_to_2nd[4][2] * (x5) + NN.intercepts_[1][2])
        return (i1, i2, i3)

    second_to_3rd = NN.coefs_[2]

    def Third_layer(x1, x2, x3):
        i1 = SD(
            second_to_3rd[0][0] * (x1) + second_to_3rd[1][0] * (x2) + second_to_3rd[2][0] * (x3) + NN.intercepts_[2][0])
        i2 = SD(
            second_to_3rd[0][1] * (x1) + second_to_3rd[1][1] * (x2) + second_to_3rd[2][1] * (x3) + NN.intercepts_[2][1])
        return i1, i2

    third_to_4th = NN.coefs_[3]

    def Fourth_layer(x1, x2):
        i1 = (third_to_4th[0][0] * (x1) + third_to_4th[1][0] * (x2) + NN.intercepts_[3][0])
        return i1

    def ActualNet(t, k1, k_1, Ca0, Ca):
        i1, i2, i3, i4, i5 = First_layer(t, k1, k_1, Ca0, Ca)
        j1, j2, j3 = Second_layer(i1, i2, i3, i4, i5)
        k1, k2 = Third_layer(j1, j2, j3)
        r = Fourth_layer(k1, k2)
        return r  # i3

    return ActualNet


# https://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
def example_scalar_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, wrt: int, num_samples=100,
                        fresh=False):
    merged_ys = [[] for _ in range(n_iter + 1)]

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, isPrecise)

    for _ in range(num_samples):
        k1_real = random.uniform(3. - k1_eps, 3. + k1_eps)
        k1_dual = 1 if wrt == 0 else 0
        k1 = Dual(k1_real, k1_dual)

        k_1_real = random.uniform(3. - k_1_eps, 3. + k_1_eps)
        k_1_dual = 1 if wrt == 1 else 0
        k_1 = Dual(k_1_real, k_1_dual)

        Ca0_real = random.uniform(1. - Ca0_eps, 1. + Ca0_eps)
        Ca0_dual = 0
        Ca0 = Dual(Ca0_real, Ca0_dual)

        y0_real = random.uniform(1. - y0_eps, 1. + y0_eps)
        y0_dual = 0
        y0 = Dual(y0_real, y0_dual)

        t0_real = 0
        t0_dual = 0
        t0 = Dual(t0_real, t0_dual)

        func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

        ys = [y0]
        ts = [t0]
        for _ in range(n_iter):
            y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
            ys.append(y_new)
            ts.append(t_new)

        for i in range(n_iter + 1):
            merged_ys[i].append(ys[i].dual)

    return merged_ys


def example_zono_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, wrt: int, fresh=False):
    # k1 = 3.0

    k1_real = Affine(Interval(3. - k1_eps, 3. + k1_eps))
    k1_dual = Affine(Interval(1, 1) if wrt == 0 else Interval(0, 0))
    k1 = Dual(k1_real, k1_dual)

    k_1_real = Affine(Interval(3. - k_1_eps, 3. + k_1_eps))
    k_1_dual = Affine(Interval(1, 1) if wrt == 1 else Interval(0, 0))
    k_1 = Dual(k_1_real, k_1_dual)

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, isPrecise)

    Ca0_real = Affine(Interval(1. - Ca0_eps, 1. + Ca0_eps))
    Ca0_dual = Affine(Interval(0, 0))
    Ca0 = Dual(Ca0_real, Ca0_dual)

    func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

    y0_real = Affine(Interval(1. - y0_eps, 1. + y0_eps))
    y0_dual = Affine(Interval(0, 0))
    y0 = Dual(y0_real, y0_dual)

    t0_real = Affine(Interval(0, 0))
    t0_dual = Affine(Interval(0, 0))
    t0 = Dual(t0_real, t0_dual)

    ys = [y0]
    ts = [t0]
    for _ in (pbar := trange(n_iter)):
        y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)
        pbar.set_description("Zonotope")

    return ys


# https://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
def example_pasado_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, wrt: int, fresh=False):
    # k1 = 3.0

    k1_real = MixedAffine(Affine(Interval(3. - k1_eps, 3. + k1_eps)))
    k1_dual = MixedAffine(Affine(Interval(1, 1) if wrt == 0 else Interval(0, 0)))
    k1 = Dual(k1_real, k1_dual)

    k_1_real = MixedAffine(Affine(Interval(3. - k_1_eps, 3. + k_1_eps)))
    k_1_dual = MixedAffine(Affine(Interval(1, 1) if wrt == 1 else Interval(0, 0)))
    k_1 = Dual(k_1_real, k_1_dual)

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, isPrecise)

    Ca0_real = MixedAffine(Affine(Interval(1. - Ca0_eps, 1. + Ca0_eps)))
    Ca0_dual = MixedAffine(Affine(Interval(0, 0)))
    Ca0 = Dual(Ca0_real, Ca0_dual)

    func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

    y0_real = MixedAffine(Affine(Interval(1. - y0_eps, 1. + y0_eps)))
    y0_dual = MixedAffine(Affine(Interval(0, 0)))
    y0 = Dual(y0_real, y0_dual)

    t0_real = MixedAffine(Affine(Interval(0, 0)))
    t0_dual = MixedAffine(Affine(Interval(0, 0)))
    t0 = Dual(t0_real, t0_dual)

    ys = [y0]
    ts = [t0]
    for _ in (pbar := trange(n_iter)):
        y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)
        pbar.set_description("Pasado")

    return ys


def example_interval(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, fresh=False):
    # k1 = 3.0

    k1_real = Interval(3. - k1_eps, 3. + k1_eps)
    k1_dual = Interval(1, 1)
    k1 = Dual(k1_real, k1_dual)

    k_1_real = Interval(3. - k_1_eps, 3. + k_1_eps)
    k_1_dual = Interval(0, 0)
    k_1 = Dual(k_1_real, k_1_dual)

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, False)

    Ca0_real = Interval(1. - Ca0_eps, 1. + Ca0_eps)
    Ca0_dual = Interval(0, 0)
    Ca0 = Dual(Ca0_real, Ca0_dual)

    func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

    y0_real = Interval(1. - y0_eps, 1. + y0_eps)
    y0_dual = Interval(0, 0)
    y0 = Dual(y0_real, y0_dual)

    t0_real = Interval(0, 0)
    t0_dual = Interval(0, 0)
    t0 = Dual(t0_real, t0_dual)

    ys = [y0]
    ts = [t0]
    for _ in range(n_iter):
        y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return [y.dual for y in ys]


# https://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
def example_zono(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, fresh=False):
    # k1 = 3.0

    k1_real = Affine(Interval(3. - k1_eps, 3. + k1_eps))
    k1_dual = Affine(Interval(1, 1))
    k1 = Dual(k1_real, k1_dual)

    k_1_real = Affine(Interval(3. - k_1_eps, 3. + k_1_eps))
    k_1_dual = Affine(Interval(0, 0))
    k_1 = Dual(k_1_real, k_1_dual)

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, isPrecise)

    Ca0_real = Affine(Interval(1. - Ca0_eps, 1. + Ca0_eps))
    Ca0_dual = Affine(Interval(0, 0))
    Ca0 = Dual(Ca0_real, Ca0_dual)

    func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

    y0_real = Affine(Interval(1. - y0_eps, 1. + y0_eps))
    y0_dual = Affine(Interval(0, 0))
    y0 = Dual(y0_real, y0_dual)

    t0_real = Affine(Interval(0, 0))
    t0_dual = Affine(Interval(0, 0))
    t0 = Dual(t0_real, t0_dual)

    ys = [y0]
    ts = [t0]
    for _ in range(n_iter):
        y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return [y.dual.interval for y in ys]


# https://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
def example_pasado(k1_eps, k_1_eps, Ca0_eps, y0_eps, isPrecise: bool, n_iter, h, fresh=False):
    #	k1 = 3.0

    k1_real = MixedAffine(Affine(Interval(3. - k1_eps, 3. + k1_eps)))
    k1_dual = MixedAffine(Affine(Interval(1, 1)))
    k1 = Dual(k1_real, k1_dual)

    k_1_real = MixedAffine(Affine(Interval(3. - k_1_eps, 3. + k_1_eps)))
    k_1_dual = MixedAffine(Affine(Interval(0, 0)))
    k_1 = Dual(k_1_real, k_1_dual)

    NN = get_nn(fresh)
    F = get_dual_lifting(NN, isPrecise)

    Ca0_real = MixedAffine(Affine(Interval(1. - Ca0_eps, 1. + Ca0_eps)))
    Ca0_dual = MixedAffine(Affine(Interval(0, 0)))
    Ca0 = Dual(Ca0_real, Ca0_dual)

    func = lambda t, Ca: F(t, k1, k_1, Ca0, Ca)

    y0_real = MixedAffine(Affine(Interval(1. - y0_eps, 1. + y0_eps)))
    y0_dual = MixedAffine(Affine(Interval(0, 0)))
    y0 = Dual(y0_real, y0_dual)

    t0_real = MixedAffine(Affine(Interval(0, 0)))
    t0_dual = MixedAffine(Affine(Interval(0, 0)))
    t0 = Dual(t0_real, t0_dual)

    ys = [y0]
    ts = [t0]
    for _ in range(n_iter):
        y_new, t_new = runge_kutta(ys[-1], ts[-1], func, h)
        ys.append(y_new)
        ts.append(t_new)

    return [y.dual.bounds for y in ys]


def test_NN():
    data, _ = generate_data()

    N = get_nn()

    F = get_dual_lifting(N)

    t = Dual(0., 0.)
    k1 = Dual(3., 1.)
    k_1 = Dual(3., 0.)
    Ca0 = Dual(1., 0)
    Ca = Dual(1., 0)

    out = F(t, k1, k_1, Ca0, Ca)
    print(out.real)
    # print(out.dual)
    print("\n")

    print(N.predict([(0, 3, 3, 1, 1)]))

    i = (0, 3, 3, 1, 1)
    print(ground_truth_f(i[0], i[1], i[2], i[3], i[4]))


def test_NN2():
    data, _ = generate_data()
    N = get_nn()
    F = get_dual_lifting(N)

    t_real = Affine(Interval(0, 0))
    t_dual = Affine(Interval(0, 0))
    t = Dual(t_real, t_dual)

    k1_real = Affine(Interval(2.8, 3.05))
    k1_dual = Affine(Interval(0, 0))
    k1 = Dual(k1_real, k1_dual)

    k_1_real = Affine(Interval(2.98, 3.05))
    k_1_dual = Affine(Interval(1, 1))
    k_1 = Dual(k_1_real, k_1_dual)

    Ca0_real = Affine(Interval(0.99, 1.05))
    Ca0_dual = Affine(Interval(0, 0))
    Ca0 = Dual(Ca0_real, Ca0_dual)

    eps = 0.0105
    y0_real = Affine(Interval(1 - eps, 1 + eps))
    y0_dual = Affine(Interval(0, 0))
    y0 = Dual(y0_real, y0_dual)

    Res = F(t, k1, k_1, Ca0, y0)
    print(Res.real.interval)
    print(Res.dual.interval)


if __name__ == '__main__':
    start = time.time()
    print(example_interval(k1_eps=0.15, k_1_eps=0.25, Ca0_eps=0.2, y0_eps=0.2, isPrecise=False, n_iter=16, h=0.025))
    print(f"Interval: {time.time() - start}s")

    start = time.time()
    print(example_zono(k1_eps=0.15, k_1_eps=0.25, Ca0_eps=0.2, y0_eps=0.2, isPrecise=False, n_iter=16, h=0.025))
    print(f"Zonotope: {time.time() - start}s")

    start = time.time()
    print(example_pasado(k1_eps=0.15, k_1_eps=0.25, Ca0_eps=0.2, y0_eps=0.2, isPrecise=True, n_iter=16, h=0.025))
    print(f"Pasado: {time.time() - start}s")

    # test_NN()
    # test_NN2()
