import numpy as np
import matplotlib.pyplot as plt
from utils import run_leapfrog, metropolis_hastings_acceptance, compute_probs, compute_energy


def hmc(prior_dist, L, epsilon, n_samples):
    samples = np.zeros(n_samples) + 1e-2
    H = np.zeros(n_samples)
    accept = 0
    kinetic_new = 1.
    for i in range(1, n_samples):
        r = prior_dist.rvs()
        if abs(kinetic_new) < 1e-7:
            theta = np.random.choice(np.linspace(1e-2, L, 51))
        else:
            theta = samples[i-1]
        theta_hat, r_hat = run_leapfrog(theta, r, L, epsilon, dist=prior_dist)
        # negate the momentum
        r_hat = -r_hat
        (potential_new, potential_old,
         kinetic_new, kinetic_old) = compute_energy(prior_dist, theta_hat,
                                                    r_hat, theta, r)
        prob_new, prob_old = compute_probs(potential_new, potential_old, kinetic_new, kinetic_old)

        if metropolis_hastings_acceptance(p_new=prob_new, p_old=prob_old):
            samples[i] = theta_hat
            accept += 1
            # plt.scatter([theta_hat], [0], c='b')
        else:
            samples[i] = samples[i-1]
        # H[i] = U(samples[i]) + K(r)
    print("accept=", accept / np.double(n_samples))
    return samples, H
