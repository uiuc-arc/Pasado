import seaborn as sns
from csv import reader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

labels = ["Interval", "Zonotope", "Pasado"]

sns.set_style('darkgrid')

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(40, 9), constrained_layout=False)


def plot(pts, ax_i):
    colors = ["#1B9E77", "#D95F02", "#7570B3"]
    markers = ["o", "^", "s"]  # circle, triangle, square

    for (x, y), i in pts:
        ax[ax_i].scatter(
            x, y, label=labels[i],
            s=50, color=colors[i], marker=markers[i], alpha=0.7
        )

    ax[ax_i].plot(np.linspace(min([t[0][0] for t in pts]), max([t[0][0] for t in pts]), 10000),
                  np.linspace(min([t[0][0] for t in pts]), max([t[0][0] for t in pts]), 10000),
                  c="red",
                  linestyle='solid')

    ax[ax_i].tick_params(axis='y', which='major', labelsize=43, pad=21)
    ax[ax_i].tick_params(axis='x', which='major', labelsize=43, pad=15)
    ax[ax_i].tick_params(axis='both', which='minor', labelsize=43, pad=21)
    ax[ax_i].set_yscale('log')
    ax[ax_i].set_xscale('log')


if __name__ == '__main__':
    pts = []
    acc_imp_int = 0.
    acc_imp_zono = 0.
    n_pts = 0
    n_int = 0
    n_zono = 0

    with open('data/black_scholes_rev_K.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[5])
            interval_u = np.float64(row[6])
            zonotope_l = np.float64(row[7])
            zonotope_u = np.float64(row[8])
            pasado_l = np.float64(row[9])
            pasado_u = np.float64(row[10])

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

    plot(pts, 0)

    pts = []

    with open('data/black_scholes_rev_S.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[5])
            interval_u = np.float64(row[6])
            zonotope_l = np.float64(row[7])
            zonotope_u = np.float64(row[8])
            pasado_l = np.float64(row[9])
            pasado_u = np.float64(row[10])

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

    plot(pts, 1)

    pts = []

    with open('data/black_scholes_rev_sigma.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[5])
            interval_u = np.float64(row[6])
            zonotope_l = np.float64(row[7])
            zonotope_u = np.float64(row[8])
            pasado_l = np.float64(row[9])
            pasado_u = np.float64(row[10])

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

    plot(pts, 2)

    pts = []

    with open('data/black_scholes_rev_tau.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[5])
            interval_u = np.float64(row[6])
            zonotope_l = np.float64(row[7])
            zonotope_u = np.float64(row[8])
            pasado_l = np.float64(row[9])
            pasado_u = np.float64(row[10])

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

    plot(pts, 3)

    pts = []

    with open('data/black_scholes_rev_r.csv', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            interval_l = np.float64(row[5])
            interval_u = np.float64(row[6])
            zonotope_l = np.float64(row[7])
            zonotope_u = np.float64(row[8])
            pasado_l = np.float64(row[9])
            pasado_u = np.float64(row[10])

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

    plot(pts, 4)

    fig.supxlabel('Pasado Interval Width', fontsize=48)
    fig.supylabel('Baseline Interval Width', fontsize=48)
    plt.subplots_adjust(left=0.07, bottom=0.18, right=1, top=0.9)

    handles, labels_ = plt.gca().get_legend_handles_labels()
    order = [0, 1]
    ax[0].legend(
        [handles[idx] for idx in order], [labels_[idx] for idx in order],
        loc="upper left",
        ncol=1,
        borderaxespad=0,
        prop={'size': 35}
    )

    print(f"Total: {n_pts}")
    print(f"Inside interval: {n_int}")
    print(f"Inside zonotope: {n_zono}")

    print(f"Average improvement over interval: {np.exp(acc_imp_int / n_pts)}")
    print(f"Average improvement over zonotope: {np.exp(acc_imp_zono / n_pts)}")
    # plt.show(dpi=700)
    fig.savefig('img/black_scholes_rev.jpg')
