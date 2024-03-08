import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from utils import (leapfrog,
                   grad,
                   run_leapfrog,
                   metropolis_hastings_acceptance,
                   compute_probs,
                   compute_energy,
                   calc_energy)


def nuts(prior_dist, L, epsilon, n_samples):
    samples = np.zeros(n_samples) + 1e-2
    H = np.zeros(n_samples)
    accept = 0
    kinetic_new = 1.
    for i in range(1, n_samples):
        r = prior_dist.rvs()
        theta = samples[i - 1]
        potential, kinetic = calc_energy(prior_dist, theta, r)
        u = np.random.uniform(low=0, high=np.exp(potential - kinetic))

        theta_minus, theta_plus = theta, theta
        r_minus, r_plus = r, r
        j = 0
        C = [(theta, r)]
        s = 1.

        while s == 1.:
            v_j = np.random.uniform(low=-1, high=1)
            if v_j < 0.:
                theta_minus, r_minus, _, _, C_prime, s_prime = build_tree(theta, r, u, v_j, j, epsilon, prior_dist)
            else:
                _, _, theta_plus, r_plus, C_prime, s_prime = build_tree(theta, r, u, v_j, j, epsilon, prior_dist)
            if s_prime == 1:
                C = C + C_prime
            s = s_prime * (1 if (theta_plus - theta_minus) * r_minus else 0) * (
                1 if (theta_plus - theta_minus) * r_plus else 0)
            j += 1
        idx = np.random.choice(list(range(len(C))))
        samples[i] = C[idx][0]
        x = [c[0] for c in C]
        y = [c[1] for c in C]
        plt.scatter(x, y)
        print(C)


def build_tree(theta, r, u, v, j, epsilon, dist):
    if j==0:
        r_prime, theta_prime = leapfrog(r,
                                        theta,
                                        epsilon,
                                        dist)
        potential, kinetic = calc_energy(dist,
                                         theta_prime,
                                         r_prime)
        energy = np.exp(potential - kinetic)
        C = [(theta_prime, r_prime)] if u <= energy else []
        delta_max = grad(theta, dist)
        s_prime = 1 if energy > np.log(u) - delta_max else 0.
        return theta_prime, r_prime, theta_prime, r_prime, C, s_prime
    else:
        theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = build_tree(theta, r, u, v, j-1, epsilon, dist)
        if v == -1:
            theta_minus, r_minus, _, _, C2_prime, s2_prime = build_tree(theta_minus,
                                                                        r_minus,
                                                                        u,
                                                                        v,
                                                                        j - 1,
                                                                        epsilon,
                                                                        dist)
        else:
            _, _, theta_plus, r_plus, C2_prime, s2_prime = build_tree(theta_plus,
                                                                      r_plus,
                                                                      u,
                                                                      v,
                                                                      j - 1,
                                                                      epsilon,
                                                                      dist)
        s_prime = s_prime*s2_prime*(1 if (theta_plus - theta_minus)*r_minus >= 0. else 0)*(1 if (theta_plus - theta_minus)*r_plus >= 0. else 0)
        C_prime = C_prime + C2_prime
        return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime
