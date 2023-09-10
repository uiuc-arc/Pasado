import argparse
import seaborn as sns
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
plt.subplots_adjust(top=1, bottom=0.2)

labels = ["Interval", "Zonotope", "Reduced Product of Zonotope and Interval"]

parser = argparse.ArgumentParser()
parser.add_argument('-l', action='store_true', help='run the example in the paper')
args = parser.parse_args()

sns.set_style('darkgrid')


def plot(pts):
    colors = ["#1B9E77", "#D95F02", "#7570B3"]
    markers = ["o", "^", "s"]  # circle, triangle, square

    for (x, y), i in pts:
        plt.scatter(
            x, y, label=labels[i],
            s=50, color=colors[i], marker=markers[i], alpha=0.7
        )

    plt.plot(np.linspace(min([t[0][0] for t in pts]), max([t[0][0] for t in pts]), 10000),
             np.linspace(min([t[0][0] for t in pts]), max([t[0][0] for t in pts]), 10000),
             c="red",
             linestyle='solid')
    plt.tick_params(axis='both', which='major', labelsize=26)
    plt.tick_params(axis='both', which='minor', labelsize=26)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pasado Interval Width', fontsize=30)
    plt.ylabel('Baseline Interval Width', fontsize=30)

    handles_, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles_))
    plt.legend(
        by_label.values(), by_label.keys(),
        loc="lower right",
        ncol=1,
        prop={'size': 22}
    )


if __name__ == '__main__':
    pts = []
    acc_imp_int = 0.
    acc_imp_zono = 0.
    threshold = 10 if args.l else 40

    n_int = 0
    n_zono = 0

    with open('data/chemical_ode_table.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[6])
            interval_u = np.float64(row[7])
            zonotope_l = np.float64(row[8])
            zonotope_u = np.float64(row[9])
            pasado_l = np.float64(row[10])
            pasado_u = np.float64(row[11])

            interval = interval_u - interval_l
            zonotope = zonotope_u - zonotope_l
            pasado = pasado_u - pasado_l

            if pasado > threshold:
                continue

            if interval <= threshold:
                n_int += 1
                acc_imp_int += np.log(interval / pasado)
                pts.append([(pasado, interval), 0])

            if zonotope <= threshold:
                n_zono += 1
                acc_imp_zono += np.log(zonotope / pasado)
                pts.append([(pasado, zonotope), 1])

    plot(pts)

    # NOTE: The statistics here are calculated for the data points in the scatter plot only,
    # which is different to the climate ODE experiment.
    print(f"Average improvement over interval: {np.exp(acc_imp_int / n_int)}")
    print(f"Average improvement over zonotope: {np.exp(acc_imp_zono / n_zono)}")
    # plt.show(dpi=700)
    plt.savefig('img/chemical_scatter.jpg')
