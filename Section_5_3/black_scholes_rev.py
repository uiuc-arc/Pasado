import itertools
import time
import gc
from tqdm import tqdm
import sys

sys.path.insert(1, '../reverse_mode_non_tensorized_src')
from interval_rev import *
from zono_rev import *
from mixed_zono_rev import *


def get_splits(l, r, n_splits):
    delta = (r - l) / n_splits
    res = []

    for i in range(n_splits):
        l_ = l + i * delta
        r_ = l_ + delta
        res.append((l_, r_))

    return res


def regular_interval(eps_K, eps_S, eps_sigma, eps_tau, eps_r):
    K = RevInterval([100. - eps_K, 100. + eps_K])
    S = RevInterval([105. - eps_S, 105. + eps_S])
    sigma = RevInterval([5. - eps_sigma, 5. + eps_sigma])
    tau = RevInterval([0.08219 - eps_tau, 0.08219 + eps_tau])
    r = RevInterval([0.0125 - eps_r, 0.0125 + eps_r])

    half = RevInterval([0.5, 0.5])
    neg_one = RevInterval([-1., -1.])

    num = i_add(
        i_log(i_div(S, K)),
        i_mul(i_add(r,
                    i_mul(half,
                          i_mul(sigma, sigma))),
              tau))

    denom = i_mul(sigma, i_sqrt(tau))

    d1 = i_div(num, denom)
    d2 = i_add(d1, i_mul(neg_one, denom))
    C_lhs = i_mul(i_normal_cdf(d1), S)
    exp_arg = i_mul(neg_one, i_mul(r, tau))
    C_rhs = i_mul(i_normal_cdf(d2), i_mul(K, i_exp(exp_arg)))
    Final = i_add(C_lhs, i_mul(neg_one, C_rhs))

    Final.backward()
    # print("###", Final)

    return K.grad, S.grad, sigma.grad, tau.grad, r.grad


def regular_interval_split(eps_K, eps_S, eps_sigma, eps_tau, eps_r, n_splits):
    def _regular_interval(K_l, K_r, S_l, S_r, sigma_l, sigma_r, tau_l, tau_r, r_l, r_r):
        K = RevInterval([K_l, K_r])
        S = RevInterval([S_l, S_r])
        sigma = RevInterval([sigma_l, sigma_r])
        tau = RevInterval([tau_l, tau_r])
        r = RevInterval([r_l, r_r])

        half = i_const(0.5)
        neg_one = i_const(-1.)

        num = i_add(
            i_log(i_div(S, K)),
            i_mul(i_add(r,
                        i_mul(half,
                              i_mul(sigma, sigma))),
                  tau))

        denom = i_mul(sigma, i_sqrt(tau))

        d1 = i_div(num, denom)
        d2 = i_add(d1, i_mul(neg_one, denom))
        C_lhs = i_mul(i_normal_cdf(d1), S)
        exp_arg = i_mul(neg_one, i_mul(r, tau))
        C_rhs = i_mul(i_normal_cdf(d2), i_mul(K, i_exp(exp_arg)))
        Final = i_add(C_lhs, i_mul(neg_one, C_rhs))

        Final.backward()
        # print("###", Final)

        return [K.grad, S.grad, sigma.grad, tau.grad, r.grad]

    K_l, K_r = 100. - eps_K, 100. + eps_K
    S_l, S_r = 105. - eps_S, 105. + eps_S
    sigma_l, sigma_r = 5. - eps_sigma, 5. + eps_sigma
    tau_l, tau_r = 0.08219 - eps_tau, 0.08219 + eps_tau
    r_l, r_r = 0.0125 - eps_r, 0.0125 + eps_r

    K = get_splits(K_l, K_r, n_splits)
    S = get_splits(S_l, S_r, n_splits)
    sigma = get_splits(sigma_l, sigma_r, 1)
    tau = get_splits(tau_l, tau_r, 1)
    r = get_splits(r_l, r_r, 1)

    args = list(itertools.product(K, S, sigma, tau, r))
    args = [list(itertools.chain(*a)) for a in args]

    res = list(map(lambda a: _regular_interval(*a), args))

    bounds = []
    for i in range(5):
        l = min([split[i].inf for split in res])
        r = max([split[i].sup for split in res])
        bounds.append(Interval(l, r))

    return bounds


def affine(eps_K, eps_S, eps_sigma, eps_tau, eps_r):
    K = RevAffine([100. - eps_K, 100. + eps_K])
    S = RevAffine([105. - eps_S, 105. + eps_S])
    sigma = RevAffine([5. - eps_sigma, 5. + eps_sigma])
    tau = RevAffine([0.08219 - eps_tau, 0.08219 + eps_tau])
    r = RevAffine([0.0125 - eps_r, 0.0125 + eps_r])

    half = RevAffine([0.5, 0.5])
    neg_one = RevAffine([-1., -1.])

    num = z_add(
        z_log(z_div(S, K)),
        z_mul(z_add(r,
                    z_mul(half,
                          z_mul(sigma, sigma))),
              tau))

    denom = z_mul(sigma, z_sqrt(tau))

    d1 = z_div(num, denom)
    d2 = z_add(d1, z_mul(neg_one, denom))
    C_lhs = z_mul(z_normal_cdf(d1), S)
    exp_arg = z_mul(neg_one, z_mul(r, tau))
    C_rhs = z_mul(z_normal_cdf(d2), z_mul(K, z_exp(exp_arg)))
    Final = z_add(C_lhs, z_mul(neg_one, C_rhs))

    Final.backward()

    return K.grad.interval, S.grad.interval, sigma.grad.interval, \
        tau.grad.interval, r.grad.interval


def affine_split(eps_K, eps_S, eps_sigma, eps_tau, eps_r, n_splits):
    def _affine(K_l, K_r, S_l, S_r, sigma_l, sigma_r, tau_l, tau_r, r_l, r_r):
        K = RevAffine([K_l, K_r])
        S = RevAffine([S_l, S_r])
        sigma = RevAffine([sigma_l, sigma_r])
        tau = RevAffine([tau_l, tau_r])
        r = RevAffine([r_l, r_r])

        half = z_const(0.5)
        neg_one = z_const(-1.)

        num = z_add(
            z_log(z_div(S, K)),
            z_mul(z_add(r,
                        z_mul(half,
                              z_mul(sigma, sigma))),
                  tau))

        denom = z_mul(sigma, z_sqrt(tau))

        d1 = z_div(num, denom)
        d2 = z_add(d1, z_mul(neg_one, denom))
        C_lhs = z_mul(z_normal_cdf(d1), S)
        exp_arg = z_mul(neg_one, z_mul(r, tau))
        C_rhs = z_mul(z_normal_cdf(d2), z_mul(K, z_exp(exp_arg)))
        Final = z_add(C_lhs, z_mul(neg_one, C_rhs))

        Final.backward()
        # print("###", Final)

        return [K.grad.interval, S.grad.interval, sigma.grad.interval,
                tau.grad.interval, r.grad.interval]

    K_l, K_r = 100. - eps_K, 100. + eps_K
    S_l, S_r = 105. - eps_S, 105. + eps_S
    sigma_l, sigma_r = 5. - eps_sigma, 5. + eps_sigma
    tau_l, tau_r = 0.08219 - eps_tau, 0.08219 + eps_tau
    r_l, r_r = 0.0125 - eps_r, 0.0125 + eps_r

    K = get_splits(K_l, K_r, n_splits)
    S = get_splits(S_l, S_r, n_splits)
    sigma = get_splits(sigma_l, sigma_r, 1)
    tau = get_splits(tau_l, tau_r, 1)
    r = get_splits(r_l, r_r, 1)

    args = list(itertools.product(K, S, sigma, tau, r))
    args = [list(itertools.chain(*a)) for a in args]

    res = list(map(lambda a: _affine(*a), args))

    bounds = []
    for i in range(5):
        l = min([split[i].inf for split in res])
        r = max([split[i].sup for split in res])
        bounds.append(Interval(l, r))

    return bounds


def affine_precise(eps_K, eps_S, eps_sigma, eps_tau, eps_r):
    K = RevAffine([100. - eps_K, 100. + eps_K])
    S = RevAffine([105. - eps_S, 105. + eps_S])
    sigma = RevAffine([5. - eps_sigma, 5. + eps_sigma])
    tau = RevAffine([0.08219 - eps_tau, 0.08219 + eps_tau])
    r = RevAffine([0.0125 - eps_r, 0.0125 + eps_r])

    half = RevAffine([0.5, 0.5])
    neg_one = RevAffine([-1., -1.])

    num = z_add(
        precise_z_log(precise_z_div(S, K)),
        precise_z_mul(z_add(r,
                            precise_z_mul(half,
                                          precise_z_mul(sigma, sigma))),
                      tau))

    denom = precise_z_mul(sigma, precise_z_sqrt(tau))

    d1 = precise_z_div(num, denom)
    d2 = z_add(d1, z_mul(neg_one, denom))
    C_lhs = precise_z_mul(precise_z_normal_cdf(d1), S)
    exp_arg = precise_z_mul(neg_one, z_mul(r, tau))
    C_rhs = precise_z_mul(precise_z_normal_cdf(d2), precise_z_mul(K, precise_z_exp(exp_arg)))
    Final = z_add(C_lhs, z_mul(neg_one, C_rhs))

    Final.backward()
    # print("###", Final)

    return K.grad.interval, S.grad.interval, sigma.grad.interval, \
        tau.grad.interval, r.grad.interval


def analytical(K, S, sigma, tau, r):
    d1 = (Log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * Sqrt(tau))
    d2 = (Log(S / K) + (r - sigma ** 2 / 2) * tau) / (sigma * Sqrt(tau))
    # C = NormalCDF(d1) * S - NormalCDF(d2) * K * Exp(-r * tau)
    # print(C)
    dK = - NormalCDF(d2) * Exp(-r * tau)
    dS = NormalCDF(d1)
    dsigma = S * NormalPDF(d1) * Sqrt(tau)
    dtau = S * NormalPDF(d1) * sigma / (2 * Sqrt(tau)) + r * K * Exp(-r * tau) * NormalCDF(d2)
    dr = K * tau * Exp(-r * tau) * NormalCDF(d2)
    return dK, dS, dsigma, dtau, dr


def mixed_affine(eps_K, eps_S, eps_sigma, eps_tau, eps_r):
    K = RevMixedAffine([100. - eps_K, 100. + eps_K])
    S = RevMixedAffine([105. - eps_S, 105. + eps_S])
    sigma = RevMixedAffine([5. - eps_sigma, 5. + eps_sigma])
    tau = RevMixedAffine([0.08219 - eps_tau, 0.08219 + eps_tau])
    r = RevMixedAffine([0.0125 - eps_r, 0.0125 + eps_r])

    half = RevMixedAffine([0.5, 0.5])
    neg_one = RevMixedAffine([-1., -1.])

    num = mixed_z_add(
        mixed_z_log(mixed_z_div(S, K)),
        mixed_z_mul(mixed_z_add(r,
                                mixed_z_mul(half,
                                            mixed_z_mul(sigma, sigma))),
                    tau))

    denom = mixed_z_mul(sigma, mixed_z_sqrt(tau))

    d1 = mixed_z_div(num, denom)
    d2 = mixed_z_add(d1, mixed_z_mul(neg_one, denom))
    C_lhs = mixed_z_mul(mixed_z_normal_cdf(d1), S)
    exp_arg = mixed_z_mul(neg_one, mixed_z_mul(r, tau))
    C_rhs = mixed_z_mul(mixed_z_normal_cdf(d2), mixed_z_mul(K, mixed_z_exp(exp_arg)))
    Final = mixed_z_add(C_lhs, mixed_z_mul(neg_one, C_rhs))

    Final.backward()
    # print("###", Final)

    return K.grad.bounds, S.grad.bounds, sigma.grad.bounds, \
        tau.grad.bounds, r.grad.bounds


def mixed_affine_precise(eps_K, eps_S, eps_sigma, eps_tau, eps_r):
    K = RevMixedAffine([100. - eps_K, 100. + eps_K])
    S = RevMixedAffine([105. - eps_S, 105. + eps_S])
    sigma = RevMixedAffine([5. - eps_sigma, 5. + eps_sigma])
    tau = RevMixedAffine([0.08219 - eps_tau, 0.08219 + eps_tau])
    r = RevMixedAffine([0.0125 - eps_r, 0.0125 + eps_r])

    half = RevMixedAffine([0.5, 0.5])
    neg_one = RevMixedAffine([-1., -1.])

    num = mixed_z_add(
        precise_mixed_z_log(precise_mixed_z_div(S, K)),
        precise_mixed_z_mul(mixed_z_add(r,
                                        precise_mixed_z_mul(half,
                                                            precise_mixed_z_mul(sigma, sigma))),
                            tau))

    denom = precise_mixed_z_mul(sigma, precise_mixed_z_sqrt(tau))

    d1 = precise_mixed_z_div(num, denom)
    d2 = mixed_z_add(d1, mixed_z_mul(neg_one, denom))
    C_lhs = precise_mixed_z_mul(precise_mixed_z_normal_cdf(d1), S)
    exp_arg = precise_mixed_z_mul(neg_one, mixed_z_mul(r, tau))
    C_rhs = precise_mixed_z_mul(precise_mixed_z_normal_cdf(d2), precise_mixed_z_mul(K, precise_mixed_z_exp(exp_arg)))
    Final = mixed_z_add(C_lhs, mixed_z_mul(neg_one, C_rhs))

    Final.backward()
    # print("###", Final)

    return K.grad.bounds, S.grad.bounds, sigma.grad.bounds, \
        tau.grad.bounds, r.grad.bounds


def mixed_affine_precise_split(eps_K, eps_S, eps_sigma, eps_tau, eps_r, n_splits):
    def _mixed_affine_precise(K_l, K_r, S_l, S_r, sigma_l, sigma_r, tau_l, tau_r, r_l, r_r):
        K = RevMixedAffine([K_l, K_r])
        S = RevMixedAffine([S_l, S_r])
        sigma = RevMixedAffine([sigma_l, sigma_r])
        tau = RevMixedAffine([tau_l, tau_r])
        r = RevMixedAffine([r_l, r_r])

        half = mixed_z_const(0.5)
        neg_one = mixed_z_const(-1.)

        num = mixed_z_add(
            precise_mixed_z_log(precise_mixed_z_div(S, K)),
            precise_mixed_z_mul(mixed_z_add(r,
                                            precise_mixed_z_mul(half,
                                                                precise_mixed_z_mul(sigma, sigma))),
                                tau))

        denom = precise_mixed_z_mul(sigma, precise_mixed_z_sqrt(tau))

        d1 = precise_mixed_z_div(num, denom)
        d2 = mixed_z_add(d1, mixed_z_mul(neg_one, denom))
        C_lhs = precise_mixed_z_mul(precise_mixed_z_normal_cdf(d1), S)
        exp_arg = precise_mixed_z_mul(neg_one, mixed_z_mul(r, tau))
        C_rhs = precise_mixed_z_mul(precise_mixed_z_normal_cdf(d2),
                                    precise_mixed_z_mul(K, precise_mixed_z_exp(exp_arg)))
        Final = mixed_z_add(C_lhs, mixed_z_mul(neg_one, C_rhs))

        Final.backward()
        # print("###", Final)

        return [K.grad.bounds, S.grad.bounds, sigma.grad.bounds,
                tau.grad.bounds, r.grad.bounds]

    K_l, K_r = 100. - eps_K, 100. + eps_K
    S_l, S_r = 105. - eps_S, 105. + eps_S
    sigma_l, sigma_r = 5. - eps_sigma, 5. + eps_sigma
    tau_l, tau_r = 0.08219 - eps_tau, 0.08219 + eps_tau
    r_l, r_r = 0.0125 - eps_r, 0.0125 + eps_r

    K = get_splits(K_l, K_r, n_splits)
    S = get_splits(S_l, S_r, n_splits)
    sigma = get_splits(sigma_l, sigma_r, 1)
    tau = get_splits(tau_l, tau_r, 1)
    r = get_splits(r_l, r_r, 1)

    args = list(itertools.product(K, S, sigma, tau, r))
    args = [list(itertools.chain(*a)) for a in args]

    res = list(map(lambda a: _mixed_affine_precise(*a), args))

    bounds = []
    for i in range(5):
        l = min([split[i].inf for split in res])
        r = max([split[i].sup for split in res])
        bounds.append(Interval(l, r))

    return bounds


if __name__ == '__main__':
    n_test = 20

    eps_K_list = [1., 5., 10.]
    eps_S_list = [1., 5., 10.]
    eps_sigma_list = [0.5, 1., 2.]
    eps_tau_list = [0.001, 0.01]
    eps_r_list = [0.001]

    configurations = []

    for eps_K in eps_K_list:
        for eps_S in eps_S_list:
            for eps_sigma in eps_sigma_list:
                for eps_tau in eps_tau_list:
                    for eps_r in eps_r_list:
                        configurations.append((eps_K, eps_S, eps_sigma, eps_tau, eps_r))

    print("TESTING configurations: ", configurations)

    total_time_interval = 0
    for config in tqdm(configurations):
        gc.collect()
        start = time.time()
        regular_interval(*config)
        end = time.time()
        total_time_interval += (end - start)
    print("Average time per configuration for Interval = ", total_time_interval / len(configurations))

    total_time_zonotope = 0
    for config in tqdm(configurations):
        Affine._weightCount = 1
        gc.collect()
        start = time.time()
        affine(*config)
        end = time.time()
        total_time_zonotope += (end - start)
    print("Average time per configuration for Zonotope = ", total_time_zonotope / len(configurations))

    total_time_pasado = 0
    for config in tqdm(configurations):
        Affine._weightCount = 1
        gc.collect()
        start = time.time()
        mixed_affine_precise(*config)
        end = time.time()
        total_time_pasado += (end - start)
    print("Average time per configuration for Pasado = ", total_time_pasado / len(configurations))
