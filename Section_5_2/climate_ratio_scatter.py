import seaborn as sns
from csv import reader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
plt.subplots_adjust(top=1, bottom=0.2)

labels = ["Interval", "Zonotope", "Pasado"]

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
    n_pts = 0
    n_int = 0
    n_zono = 0

    with open('data/climate_ode_table.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[7])
            interval_u = np.float64(row[8])
            zonotope_l = np.float64(row[9])
            zonotope_u = np.float64(row[10])
            pasado_l = np.float64(row[11])
            pasado_u = np.float64(row[12])

            interval = interval_u - interval_l
            zonotope = zonotope_u - zonotope_l
            pasado = pasado_u - pasado_l

            if interval_l <= pasado_l <= pasado_u <= interval_u:
                n_int += 1

            if zonotope_l <= pasado_l <= pasado_u <= zonotope_u:
                n_zono += 1

            pts.append([(pasado, interval), 0])
            pts.append([(pasado, zonotope), 1])

            acc_imp_int += np.log(interval / pasado)
            acc_imp_zono += np.log(zonotope / pasado)
            n_pts += 1

    plot(pts)

    print(f"Total: {n_pts}")
    print(f"Inside interval: {n_int}")
    print(f"Inside zonotope: {n_zono}")

    print(f"Average improvement over interval: {np.exp(acc_imp_int / n_pts)}")
    print(f"Average improvement over zonotope: {np.exp(acc_imp_zono / n_pts)}")
    # plt.show(dpi=700)
    plt.savefig('img/climate_scatter.jpg')
