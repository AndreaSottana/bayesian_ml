import numpy as np
from matplotlib import pyplot as plt
from src.em import *


if __name__ == '__main__':
    samples = np.load('../data/samples.npz')
    X_ = samples['data']
    pi0_ = samples['pi0']
    mu0_ = samples['mu0']
    sigma0_ = samples['sigma0']
    gamma_ = e_step(X_, pi0_, mu0_, sigma0_)
    gamma_optimised = e_step_optimized(X_, pi0_, mu0_, sigma0_)
    np.testing.assert_allclose(gamma_, gamma_optimised)
    pi_, mu_, sigma_ = m_step(X_, gamma_)
    pi_optimised, mu_optimised, sigma_optimised = m_step_optimized(X_, gamma_optimised)
    np.testing.assert_allclose(pi_, pi_optimised)
    np.testing.assert_allclose(mu_, mu_optimised)
    np.testing.assert_allclose(sigma_, sigma_optimised)
    loss_ = compute_vlb(X_, pi_, mu_, sigma_, gamma_)
    loss_optimised = compute_vlb_optimized(X_, pi_optimised, mu_optimised, sigma_optimised, gamma_optimised)
    np.testing.assert_allclose(loss_, loss_optimised)
    best_loss, best_pi, best_mu, best_sigma = train_EM(X_, C=3)
    gamma = e_step_optimized(X_, best_pi, best_mu, best_sigma)
    labels = gamma.argmax(axis=1)
    colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.
    plt.scatter(X_[:, 0], X_[:, 1], c=colors[labels], s=30)
    plt.axis('equal')
    plt.show()
