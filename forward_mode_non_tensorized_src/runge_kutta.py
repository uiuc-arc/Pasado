from dual_intervals import *
import matplotlib.pyplot as plt


def plot_derivs(ts, duals):
    plt.plot([t.real for t in ts], [x.dual for x in duals])
    plt.ylabel('Sensitivity')
    plt.show()


def runge_kutta(yn, tn, f, h):
    tn_plus1 = tn + h

    k1 = f(tn, yn) * h

    k2 = f((tn + (0.5 * h)), yn + (k1 * 0.5)) * h

    k3 = f((tn + (0.5 * h)), yn + (k2 * 0.5)) * h

    k4 = f((tn + h), yn + (k3)) * h

    yn_plus1 = yn + (((k1) + (2. * k2) + (2. * k3) + (k4)) * (1. / 6.))

    return (yn_plus1, tn_plus1)


def euler(yn, tn, func, h):
    yn_plus_one = yn + (h * func(tn, yn))
    tn_plus_one = tn + h
    return (yn_plus_one, tn_plus_one)
