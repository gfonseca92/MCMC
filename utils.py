import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def prob_function(dist, theta, r):
    f_theta = dist.pdf(theta)
    f_momentum = r*r*0.5
    f = np.exp(f_theta - f_momentum)
    return f


def grad(theta, dist):
    eps = 1e-9
    return (dist.logpdf(theta + eps) - dist.logpdf(theta - eps)) / (2*eps)


def leapfrog(r, theta, epsilon, dist):
    if np.random.rand() > 0.5:
        r += epsilon * grad(theta, dist)/2
        theta += epsilon * r
        r += epsilon * grad(theta, dist)/2
    else:
        r -= epsilon * grad(theta, dist) / 2
        theta -= epsilon * r
        r -= epsilon * grad(theta, dist) / 2
    return r, theta


def run_leapfrog(theta, r, L, epsilon, dist):
    steps = int(L / epsilon)
    r_vec, theta_vec = np.zeros(steps), np.zeros(steps)
    for s in range(steps):
        r, theta = leapfrog(r, theta, epsilon, dist)
        r_vec[s] = r
        theta_vec[s] = theta
    # if np.random.rand() < 0.10:
    #     plt.plot(theta_vec, r_vec, '--k')
    #     plt.scatter([theta], [r], c='k')
    return theta, r


def metropolis_hastings_acceptance(p_new, p_old):
    alpha = min(1, p_new / p_old)
    return np.random.uniform() < alpha


def compute_probs(potential_new, potential_old, kinetic_new, kinetic_old):
    prob_new = np.exp(potential_new - kinetic_new)
    prob_old = np.exp(potential_old - kinetic_old)
    return prob_new, prob_old


def calc_energy(dist: st, theta, r):
    return dist.logpdf(theta), 0.5 * r ** 2


def compute_energy(dist: st, theta_hat, r_hat, theta, r):
    potential_old, kinetic_old = calc_energy(dist, theta, r)
    potential_new, kinetic_new = calc_energy(dist, theta_hat, r_hat)
    return potential_new, potential_old, kinetic_new, kinetic_old
