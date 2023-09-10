import gc
import time
import random
import cProfile
import pstats

import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from tqdm import trange

import sys

sys.path.insert(0, '../forward_mode_non_tensorized_src')
sys.path.insert(0, '../reverse_mode_non_tensorized_src')
from interval_rev import *
from zono_rev import *
from mixed_zono_rev import *

continuous_idx = {0, 2, 3, 4, 5}


def adult_interval(input: Sequence, eps: Sequence, cls: MLPClassifier, return_count=False):
    if len(eps) != len(input):
        raise ValueError("Input sizes don't match!")

    hidden_ = [np.vectorize(i_const)(x) for x in cls.coefs_]
    biases_ = [np.vectorize(i_const)(x) for x in cls.intercepts_]

    x = np.array([RevInterval([input[i_] - eps[i_], input[i_] + eps[i_]]) for i_ in range(len(input))], dtype=object)
    h = [x]
    for i_ in range(len(hidden_)):
        h.append(h[-1] @ hidden_[i_] + biases_[i_])
        h.append(np.vectorize(i_tanh)(h[-1])
                 if i_ != len(hidden_) - 1 else np.vectorize(i_sigmoid)(h[-1]))

    # print("Interval real: ", h[-1][0])  # Final result real part.

    h[-1][0].backward()

    res = [x[i_].grad for i_ in continuous_idx]
    count = tuple(x_ >= 0. for x_ in res), tuple(x_ <= 0. for x_ in res)
    return count if return_count else res


def adult_affine(input: Sequence, eps: Sequence, cls: MLPClassifier, return_count=False):
    if len(eps) != len(input):
        raise ValueError("Input sizes don't match!")

    Affine._weightCount = 1  # Otherwise the code gets slower over time.

    hidden_ = [np.vectorize(z_const)(x) for x in cls.coefs_]
    biases_ = [np.vectorize(z_const)(x) for x in cls.intercepts_]

    x = np.array([RevAffine([input[i_] - eps[i_], input[i_] + eps[i_]]) for i_ in range(len(input))], dtype=object)
    h = [x]
    for i_ in range(len(hidden_)):
        h.append(h[-1] @ hidden_[i_] + biases_[i_])
        h.append(np.vectorize(z_tanh)(h[-1])
                 if i_ != len(hidden_) - 1 else np.vectorize(z_sigmoid)(h[-1]))

    # print("Affine real: ", h[-1][0])  # Final result real part.

    # dot = draw_dot(h[-1][0])
    # dot.render('gout')
    # print("Done")

    h[-1][0].backward()

    res = [x[i_].grad.interval for i_ in continuous_idx]
    count = tuple(x_ >= 0. for x_ in res), tuple(x_ <= 0. for x_ in res)
    return count if return_count else res


def adult_pasado(input: Sequence, eps: Sequence, cls: MLPClassifier, return_count=False):
    if len(eps) != len(input):
        raise ValueError("Input sizes don't match!")

    Affine._weightCount = 1  # Otherwise the code gets slower over time.

    hidden_ = [np.vectorize(mixed_z_const)(x) for x in cls.coefs_]
    biases_ = [np.vectorize(mixed_z_const)(x) for x in cls.intercepts_]

    x = np.array([RevMixedAffine([input[i_] - eps[i_], input[i_] + eps[i_]]) for i_ in range(len(input))], dtype=object)
    h = [x]
    for i_ in range(len(hidden_)):
        h.append(h[-1] @ hidden_[i_] + biases_[i_])
        h.append(np.vectorize(precise_mixed_z_tanh)(h[-1])
                 if i_ != len(hidden_) - 1 else np.vectorize(precise_mixed_z_sigmoid)(h[-1]))

    # print("Pasado real: ", h[-1][0])  # Final result real part.

    h[-1][0].backward()

    res = [x[i_].grad.bounds for i_ in continuous_idx]
    count = tuple(x_ >= 0. for x_ in res), tuple(x_ <= 0. for x_ in res)
    return count if return_count else res


def main_execution(n_times=20):
    """
    Only used to calculate average runtimes.
    """
    for _ in trange(n_times):
        inputs = random.choice(all_inputs)
        eps_value = random.choice(eps_values)

        eps = [0.] * len(inputs)
        for i in continuous_idx:
            inputs[i] = 0.
            eps[i] = eps_value

        start = time.time()
        adult_interval(inputs, eps, cls, return_count=return_count)
        interval_times.append(time.time() - start)

        start = time.time()
        adult_affine(inputs, eps, cls, return_count=return_count)
        affine_times.append(time.time() - start)

        gc.collect()

        adult_pasado(inputs, eps, cls, return_count=return_count)
        pasado_times.append(time.time() - start)

        gc.collect()

    print(f"Average Interval time = {sum(interval_times) / len(interval_times)}s")
    print(f"Average Affine time = {sum(affine_times) / len(affine_times)}s")
    print(f"Average Pasado time = {sum(pasado_times) / len(pasado_times)}s")


if __name__ == '__main__':
    all_inputs = pickle.load(open('saved/inputs.sav', 'rb'))
    cls = pickle.load(open('saved/cls.sav', 'rb'))

    eps_values = [i * 0.02 for i in range(20)]  # 0.0 to 0.4
    return_count = False

    interval_times = []
    affine_times = []
    pasado_times = []

    main_execution()
