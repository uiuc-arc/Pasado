from multiprocessing import Pool, Manager, cpu_count
import csv
import warnings

from tabulate import tabulate
import numpy as np
from tqdm import tqdm

from climate_ode_v2 import climate_interval, climate_regular_zono, climate_mixed_precise

headers = ["y0", "R", "Q", "alpha", "sigma", "h", "n",
           "Interval", "Zonotope", "Pasado"]


def example_wrapper(arg, table):
    y0, R, Q, alpha, sigma, h, iter_list = arg
    n = max(iter_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = climate_interval(y0[0], y0[1], R[0], R[1], Q[0], Q[1], alpha[0], alpha[1], sigma[0], sigma[1], h, n)
        r2 = climate_regular_zono(y0[0], y0[1], R[0], R[1], Q[0], Q[1], alpha[0], alpha[1], sigma[0], sigma[1], h, n)
        r3 = climate_mixed_precise(y0[0], y0[1], R[0], R[1], Q[0], Q[1], alpha[0], alpha[1], sigma[0], sigma[1], h, n)

    r1 = [r.dual for r in r1]
    r2 = [r.dual.interval for r in r2]
    r3 = [r.dual.bounds for r in r3]

    for n in iter_list:
        table.append([y0, R, Q, alpha, sigma, h, n,
                      r1[n], r2[n], r3[n]])  # `append()` should be thread-safe.


def example_wrapper_star(args):
    example_wrapper(*args)


y0_list = [(300, 375), (275, 400)]
R_list = [(2.65, 2.95)]
Q_list = [(342, 342), (270, 450)]
alpha_list = [(0.35, 0.35), (0.3, 0.35)]
sigma_list = [(5.67037442e-8 * 0.6, 5.67037442e-8 * 0.9)]
iter_list = [8, 12]
h_list = [0.025, 0.05]

if __name__ == '__main__':

    num_cpus = cpu_count() - 2 if cpu_count() - 2 > 0 else 1

    args = list()
    for y0 in y0_list:
        for R in R_list:
            for Q in Q_list:
                for alpha in alpha_list:
                    for sigma in sigma_list:
                        for h in h_list:
                            args.append((y0, R, Q, alpha, sigma, h, iter_list))

    manager = Manager()
    table = manager.list([])
    args = [(arg, table) for arg in args]
    with Pool(num_cpus) as p:
        _ = list(tqdm(p.imap(example_wrapper_star, args), total=len(args)))

    table = list(table)  # Retrieve the shared list.
    table.sort()
    print("Evaluation complete.")

    with open("data/climate_ode_table.html", "w") as html, open("data/climate_ode_table.csv", "w", newline="") as f:
        print(tabulate(table, headers, tablefmt="html", stralign="center"), file=html)
        transformed = list()
        for row in table:
            temp = row[0:7]
            temp.extend(map(np.float64, [row[7].inf, row[7].sup, row[8].inf, row[8].sup,
                                         row[9].inf, row[9].sup]))
            transformed.append(temp)
        wr = csv.writer(f)
        wr.writerows(transformed)
