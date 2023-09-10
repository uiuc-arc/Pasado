import os
from multiprocessing import Pool, Manager, cpu_count
import csv

from tabulate import tabulate
import numpy as np
from tqdm import tqdm

from black_scholes_rev import regular_interval, affine, mixed_affine_precise

headers = ["eps_K", "eps_S", "eps_sigma", "eps_tau", "eps_r", "Interval", "Zonotope", "Pasado"]


def wrapper(arg, table):
    eps_K, eps_S, eps_sigma, eps_tau, eps_r = arg
    r1 = regular_interval(eps_K, eps_S, eps_sigma, eps_tau, eps_r)
    r2 = affine(eps_K, eps_S, eps_sigma, eps_tau, eps_r)
    r3 = mixed_affine_precise(eps_K, eps_S, eps_sigma, eps_tau, eps_r)
    row = [eps_K, eps_S, eps_sigma, eps_tau, eps_r]
    row.extend(r1 + r2 + r3)
    table.append(row)


def wrapper_star(args):
    wrapper(*args)


# eps_K_list = [1.]
# eps_S_list = [1.]
# eps_sigma_list = [2.0]
# eps_tau_list = [0.01]
# eps_r_list = [0.001]

eps_K_list = [1., 5., 10.]
eps_S_list = [1., 5., 10.]
eps_sigma_list = [0.5, 1., 2.]
eps_tau_list = [0.001, 0.01]
eps_r_list = [0.001]

if __name__ == '__main__':

    num_cpus = cpu_count() - 2 if cpu_count() - 2 > 0 else 1

    args = list()
    for eps_K in eps_K_list:
        for eps_S in eps_S_list:
            for eps_sigma in eps_sigma_list:
                for eps_tau in eps_tau_list:
                    for eps_r in eps_r_list:
                        args.append((eps_K, eps_S, eps_sigma, eps_tau, eps_r))

    manager = Manager()
    table = manager.list([])
    args = [(arg, table) for arg in args]
    with Pool(num_cpus) as p:
        _ = list(tqdm(p.imap(wrapper_star, args), total=len(args)))

    table = list(table)  # Retrieve the shared list.
    table.sort()
    print("Done evaluation.")

    path_prefix = "data/black_scholes_rev_"

    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    with open(path_prefix + "K.html", "w") as K_html, \
            open(path_prefix + "K.csv", "w", newline="") as K_csv, \
            open(path_prefix + "S.html", "w") as S_html, \
            open(path_prefix + "S.csv", "w", newline="") as S_csv, \
            open(path_prefix + "sigma.html", "w") as sigma_html, \
            open(path_prefix + "sigma.csv", "w", newline="") as sigma_csv, \
            open(path_prefix + "tau.html", "w") as tau_html, \
            open(path_prefix + "tau.csv", "w", newline="") as tau_csv, \
            open(path_prefix + "r.html", "w") as r_html, \
            open(path_prefix + "r.csv", "w", newline="") as r_csv:
        print(tabulate([row[:5] + [row[i] for i in (5, 10, 15)] for row in table],
                       headers, tablefmt="html", stralign="center"), file=K_html)
        print(tabulate([row[:5] + [row[i] for i in (6, 11, 16)] for row in table],
                       headers, tablefmt="html", stralign="center"), file=S_html)
        print(tabulate([row[:5] + [row[i] for i in (7, 12, 17)] for row in table],
                       headers, tablefmt="html", stralign="center"), file=sigma_html)
        print(tabulate([row[:5] + [row[i] for i in (8, 13, 18)] for row in table],
                       headers, tablefmt="html", stralign="center"), file=tau_html)
        print(tabulate([row[:5] + [row[i] for i in (9, 14, 19)] for row in table],
                       headers, tablefmt="html", stralign="center"), file=r_html)

        rows = [[] for _ in range(5)]
        for row in table:
            for i in range(5):
                temp = row[:5]
                temp.extend(map(np.float64, [row[5 + i].inf, row[5 + i].sup, row[10 + i].inf, row[10 + i].sup,
                                             row[15 + i].inf, row[15 + i].sup]))
                rows[i].append(temp)

        wr = csv.writer(K_csv)
        wr.writerows(rows[0])
        wr = csv.writer(S_csv)
        wr.writerows(rows[1])
        wr = csv.writer(sigma_csv)
        wr.writerows(rows[2])
        wr = csv.writer(tau_csv)
        wr.writerows(rows[3])
        wr = csv.writer(r_csv)
        wr.writerows(rows[4])

    print("Done table-filling.")
