import time
import csv
import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count
import pickle

from tqdm import tqdm
from tabulate import tabulate

from adult_eval import adult_interval, adult_affine, adult_pasado, continuous_idx

headers = [
    "eps", "Interval # (↑, ↓, ↑ + ↓)",
    "Zonotope # (↑, ↓, ↑ + ↓)", "Pasado # (↑, ↓, ↑ + ↓)"
]


def wrapper(args):
    id, inputs, eps, cls = args

    eps_ = [0.] * len(inputs)

    for i_ in continuous_idx:
        inputs[i_] = 0.  # Mean.
        eps_[i_] = eps

    int_i, int_d = adult_interval(inputs, eps_, cls, return_count=True)

    aff_i, aff_d = adult_affine(inputs, eps_, cls, return_count=True)

    pas_i, pas_d = adult_pasado(inputs, eps_, cls, return_count=True)

    return eps, (
        (sum(int_i), sum(int_d), sum(int_i) + sum(int_d)),
        (sum(aff_i), sum(aff_d), sum(aff_i) + sum(aff_d)),
        (sum(pas_i), sum(pas_d), sum(pas_i) + sum(pas_d))
    )


def generate_args(inputs, cls, eps_list):
    for eps_ in eps_list:
        for id_, inputs_ in enumerate(inputs):
            yield id_, inputs_, eps_, cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='run the full experiment')
    params = parser.parse_args()

    num_cpus = max(cpu_count() - 2, 1)
    print(f"Using {num_cpus} CPU cores.")

    if params.l:
        eps_list = np.arange(0.0, 1.001, 0.01)
    else:
        eps_list = np.arange(0.0, 0.501, 0.05)  # A coarser range of epsilon.

    print("Input `eps`: ", eps_list)

    with open('saved/inputs.sav', 'rb') as f:
        inputs = pickle.load(f)
    with open('saved/cls.sav', 'rb') as f:
        cls = pickle.load(f)

    if not params.l:
        inputs = inputs[:5]  # A subset of inputs.

    eps_aggregate = {}
    with Pool(num_cpus) as p:
        for eps, values in tqdm(p.imap_unordered(wrapper, generate_args(inputs, cls, eps_list)),
                                total=len(inputs) * len(eps_list)):
            if eps not in eps_aggregate:
                eps_aggregate[eps] = [np.array(val) for val in values]
            else:
                for i, val in enumerate(values):
                    eps_aggregate[eps][i] += np.array(val)

    results = [(eps, *map(tuple, vals)) for eps, vals in sorted(eps_aggregate.items())]
    print("Done evaluation.")

    os.makedirs('results', exist_ok=True)

    with open("results/adult.html", "w", encoding="utf-8") as html, open("results/adult.csv", "w", newline="") as f:
        html.writelines('<meta charset="UTF-8">\n')
        table_str = tabulate(results, headers, tablefmt="html", stralign="center")
        html.write(table_str)
        wr = csv.writer(f)
        wr.writerows(results)

    print("Done table-filling.")
