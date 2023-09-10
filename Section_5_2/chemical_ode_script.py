from multiprocessing import Pool, Manager, cpu_count
import csv
import argparse

from tabulate import tabulate
import numpy as np
from tqdm import tqdm

from chemical_example import example_zono, example_pasado, example_interval

headers = ["k1_eps", "k_1_eps", "Ca0_eps", "y0_eps", "n_iter", "h", "Interval", "Zonotope",
           "Pasado"]

parser = argparse.ArgumentParser()
parser.add_argument('-l', action='store_true', help='run the example in the paper')
parser.add_argument('-f', action='store_true', help='train a fresh network')
params = parser.parse_args()


def example_wrapper(arg, table):
    k1_eps, k_1_eps, Ca0_eps, y0_eps, iter_list, h = arg
    n_iter = max(iter_list)
    r1 = example_interval(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, params.f)
    r2 = example_zono(k1_eps, k_1_eps, Ca0_eps, y0_eps, False, n_iter, h, params.f)
    r3 = example_pasado(k1_eps, k_1_eps, Ca0_eps, y0_eps, True, n_iter, h, params.f)
    for n in iter_list:
        table.append([k1_eps, k_1_eps, Ca0_eps, y0_eps, n, h,
                      r1[n], r2[n], r3[n]])  # `append()` should be thread-safe.


def example_wrapper_star(args):
    example_wrapper(*args)


if params.l:
    k1_eps_list = [0.05, 0.1, 0.15, 0.2]
    k_1_eps_list = [0.1, 0.2, 0.25, 0.3]
    Ca0_eps_list = [0.1, 0.2]
    y0_eps_list = [0.1, 0.2]
    iter_list = [8, 10, 12, 16]
    h_list = [0.025, 0.1]
else:
    k1_eps_list = [0.05]
    k_1_eps_list = [0.1, 0.2]
    Ca0_eps_list = [0.1, 0.2]
    y0_eps_list = [0.1, 0.2]
    iter_list = [8, 10]
    h_list = [0.025]

if __name__ == '__main__':

    num_cpus = cpu_count() - 2 if cpu_count() - 2 > 0 else 1

    args = list()
    for k1_eps in k1_eps_list:
        for k_1_eps in k_1_eps_list:
            for Ca0_eps in Ca0_eps_list:
                for y0_eps in y0_eps_list:
                    for h in h_list:
                        args.append((k1_eps, k_1_eps, Ca0_eps, y0_eps, iter_list, h))

    manager = Manager()
    table = manager.list([])
    args = [(arg, table) for arg in args]
    with Pool(num_cpus) as p:
        _ = list(tqdm(p.imap(example_wrapper_star, args), total=len(args)))

    table = list(table)  # Retrieve the shared list.
    table.sort()
    print("Evaluation complete.")

    with open("data/chemical_ode_table.html", "w") as html, open("data/chemical_ode_table.csv", "w", newline="") as f:
        print(tabulate(table, headers, tablefmt="html", stralign="center"), file=html)
        transformed = list()
        for row in table:
            temp = row[0:6]
            temp.extend(map(np.float64, [row[6].inf, row[6].sup, row[7].inf, row[7].sup, row[8].inf, row[8].sup]))
            transformed.append(temp)
        wr = csv.writer(f)
        wr.writerows(transformed)
