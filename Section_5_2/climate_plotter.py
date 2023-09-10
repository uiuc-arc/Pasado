import time
import random
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from climate_ode_v2 import climate_scalar, climate_interval, climate_regular_zono, climate_mixed_precise

colors = ["#1B9E77", "#D95F02", "#7570B3", "#4C72B0"]


def plot(bounds_interval, bounds_zono, bounds_precise, scalar):
    if not len(bounds_interval) == len(bounds_zono) == len(bounds_precise) == len(scalar):
        raise ValueError("Input size does not match!")

    labels = [f"${{{i + 1}}}$" for i in range(len(bounds_zono))]

    width = 0.4
    sns.set_style('darkgrid')

    for i in range(len(labels)):
        plt.plot([i - width / 2., i + width / 2.], [bounds_interval[i][0], bounds_interval[i][0]], color=colors[0],
                 label="Interval" if i == 0 else None)
        plt.plot([i - width / 2., i + width / 2.], [bounds_interval[i][1], bounds_interval[i][1]], color=colors[0])
        plt.plot([i - width / 2., i + width / 2.], [bounds_zono[i][0], bounds_zono[i][0]], color=colors[1],
                 label="Zonotope" if i == 0 else None)
        plt.plot([i - width / 2., i + width / 2.], [bounds_zono[i][1], bounds_zono[i][1]], color=colors[1])
        plt.plot([i - width / 2., i + width / 2.], [bounds_precise[i][0], bounds_precise[i][0]],
                 color=colors[2], label="Pasado" if i == 0 else None)
        plt.plot([i - width / 2., i + width / 2.], [bounds_precise[i][1], bounds_precise[i][1]], color=colors[2])

        for j, point in enumerate(scalar[i]):
            x_coordinate = random.uniform(i - width / 2., i + width / 2.)
            plt.scatter(x_coordinate, point, color=colors[3], label="Scalar" if i == j == 0 else None, s=1)

    plt.axhline(y=0., color='k', linestyle=':')
    plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Time Step", fontsize=18)
    plt.ylabel("Sensitivity", fontsize=18)
    plt.legend(fontsize=13)

    # plt.show()
    plt.savefig(f'img/climate_step.jpg', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    # The initial value is not shown here.
    start = time.time()
    r_interval = climate_interval(275, 400, 2.65, 2.95, 342, 342, 0.35, 0.35, 5.67037442e-8 * 0.6, 5.67037442e-8 * 0.9,
                                  0.025, 12)[1:]
    print(f"Interval: {time.time() - start}s")

    start = time.time()
    r_zono = climate_regular_zono(275, 400, 2.65, 2.95, 342, 342, 0.35, 0.35, 5.67037442e-8 * 0.6, 5.67037442e-8 * 0.9,
                                  0.025, 12)[1:]
    print(f"Zonotope: {time.time() - start}s")

    start = time.time()
    r_pasado = climate_mixed_precise(275, 400, 2.65, 2.95, 342, 342, 0.35, 0.35, 5.67037442e-8 * 0.6,
                                     5.67037442e-8 * 0.9, 0.025, 12)[1:]
    print(f"Pasado: {time.time() - start}s")

    scalar = climate_scalar(275, 400, 2.65, 2.95, 342, 342, 0.35, 0.35, 5.67037442e-8 * 0.6, 5.67037442e-8 * 0.9,
                            0.025, 12, num_samples=200)[1:]

    bounds_interval = [[x.dual.inf, x.dual.sup] for x in r_interval]
    bounds_zono = [[x.dual.interval.inf, x.dual.interval.sup] for x in r_zono]
    bounds_pasado = [[x.dual.bounds.inf, x.dual.bounds.sup] for x in r_pasado]

    plot(bounds_interval, bounds_zono, bounds_pasado, scalar)
