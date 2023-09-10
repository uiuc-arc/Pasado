import random
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from chemical_example import example_scalar_step, example_zono_step, example_pasado_step

parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store_true', help='train a fresh network')
args = parser.parse_args()

sns.set_style('darkgrid')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), constrained_layout=False)
plt.subplots_adjust(top=1)

colors = ["#1B9E77", "#D95F02", "#7570B3", "#4C72B0"]


def plot(bounds_zono, bounds_precise, scalar, ax_i):
    if len(bounds_zono) != len(bounds_precise):
        raise ValueError("Input size does not match!")

    labels = [f"${{{i + 1}}}$" for i in range(len(bounds_zono))]

    width = 0.4
    # fig, ax = plt.subplots()

    for i in range(len(labels)):
        ax[ax_i].plot([i - width / 2., i + width / 2.], [bounds_zono[i][0], bounds_zono[i][0]], color=colors[1],
                      label="Zonotope" if i == 0 else None)
        ax[ax_i].plot([i - width / 2., i + width / 2.], [bounds_zono[i][1], bounds_zono[i][1]], color=colors[1])
        ax[ax_i].plot([i - width / 2., i + width / 2.], [bounds_precise[i][0], bounds_precise[i][0]],
                      color=colors[2], label="Pasado" if i == 0 else None)
        ax[ax_i].plot([i - width / 2., i + width / 2.], [bounds_precise[i][1], bounds_precise[i][1]], color=colors[2])

        for j, point in enumerate(scalar[i]):
            x_coordinate = random.uniform(i - width / 2., i + width / 2.)
            ax[ax_i].scatter(x_coordinate, point, color=colors[3], label="Scalar" if i == j == 0 else None, s=1)

    if ax_i == 0:
        ax[ax_i].axhline(y=0., color='k', linestyle=':')
        ax[ax_i].legend(loc='lower left', fontsize=20)
    ax[ax_i].set_xticks(ticks=range(len(labels)), labels=labels)
    ax[ax_i].tick_params(axis='both', which='major', labelsize=22)
    ax[ax_i].tick_params(axis='both', which='minor', labelsize=22)


k1_eps = 0.05
k_1_eps = 0.2
Ca0_eps = 0.2
y0_eps = 0.1
n_iter = 16
h = 0.025

if __name__ == '__main__':
    n_samples = 100
    # The initial value is not shown here.
    scalar = example_scalar_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, 0, n_samples, args.f)[1:]
    r_zono = example_zono_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, 0, args.f)[1:]
    r_pasado = example_pasado_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, True, n_iter, h, 0, args.f)[1:]

    bounds_zono = [[x.dual.interval.inf, x.dual.interval.sup] for x in r_zono]
    bounds_pasado = [[x.dual.bounds.inf, x.dual.bounds.sup] for x in r_pasado]

    plot(bounds_zono, bounds_pasado, scalar, 0)

    scalar = example_scalar_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, 1, n_samples, args.f)[1:]
    r_zono = example_zono_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, 1, args.f)[1:]
    r_pasado = example_pasado_step(k1_eps, k_1_eps, Ca0_eps, y0_eps, True, n_iter, h, 1, args.f)[1:]

    bounds_zono = [[x.dual.interval.inf, x.dual.interval.sup] for x in r_zono]
    bounds_pasado = [[x.dual.bounds.inf, x.dual.bounds.sup] for x in r_pasado]

    plot(bounds_zono, bounds_pasado, scalar, 1)

    fig.supxlabel("Time Step", fontsize=28)
    fig.supylabel("Sensitivity", fontsize=28)
    plt.subplots_adjust(left=0.1)

    # plt.show()
    fig.savefig(f'img/chemical_step.jpg')
