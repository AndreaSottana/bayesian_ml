import numpy as np


def e_step(x: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Performs the E-step (expectation step) on the Gaussian Mixture Model (GMM).
    Each input is numpy array. Shapes are presented below, where
    - N: number of data points
    - d: number of dimensions
    - C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)
    :param x: (N x d), input data points
    :param pi: (C), mixture component weights
    :param mu: (C x d), mixture component means
    :param sigma: (C x d x d), mixture component covariance matrices
    :return: gamma: (N x C), probabilities of clusters for objects; each component along axis = 0 [0, N-1] represents
             a point, each component along axis = 1 [0, C-1] represent the probability of that point belonging to
             cluster c={0, 1, ..., C-2, C-1}
    """
    assert pi.shape[0] == mu.shape[0] == sigma.shape[0], "Some shapes are incompatible"
    assert x.shape[1] == mu.shape[1] == sigma.shape[1] == sigma.shape[2], "Some shapes are incompatible"
    N = x.shape[0]
    C = pi.shape[0]
    d = x.shape[1]
    gamma = np.zeros((N, C))  # distribution q(T)

    for i in range(N):
        weighted_gaussians = np.zeros(C)
        for c in range(C):
            norm_coeff = (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma[c])))
            gaussian = norm_coeff * np.exp(
                -0.5 * np.matmul(np.transpose(x[i] - mu[c]), np.linalg.solve(sigma[c], x[i] - mu[c]))
            )
            weighted_gaussian = pi[c] * gaussian
            weighted_gaussians[c] = weighted_gaussian
        gamma[i] = weighted_gaussians / np.sum(weighted_gaussians)

    return gamma


def e_step_optimized(x: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Same as e_step (refer to e_step for documentation), but with better performance optimization due to
    broadcasting for speed and other tricks to improve numerical stability when calculating large / small numbers.
    Some common factors, e.g. 1 / np.sqrt(np.power(2 * np.pi, d) have also been removed as they would be
    cancelled out in normalization operations.
    """
    assert pi.shape[0] == mu.shape[0] == sigma.shape[0], "Some shapes are incompatible"
    assert x.shape[1] == mu.shape[1] == sigma.shape[1] == sigma.shape[2], "Some shapes are incompatible"

    gaussians = np.einsum(
        'ijkl, ijkl -> ij',
        (x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
        np.linalg.solve(sigma, (x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis])
    )
    gaussians = gaussians - np.max(gaussians, axis=1)[:, np.newaxis]  # trick for numerical stability
    gaussians = np.exp(-0.5 * gaussians)
    weighted_gaussians = pi * gaussians / np.sqrt(np.linalg.det(sigma))
    gamma = weighted_gaussians / np.sum(weighted_gaussians, axis=1)[:, np.newaxis]
    return gamma


def m_step(x, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    x: (N x d), data points
    gamma: (N x C), distribution q(T)

    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = x.shape[0]  # number of objects
    C = gamma.shape[1]  # number of clusters
    d = x.shape[1]  # dimension of each object

    pi = gamma.sum(axis=0) / N
    mu = np.zeros((C, d))
    for c in range(C):
        mu[c] = np.dot(gamma[:, c], x) / gamma.sum(axis=0)[c]

    sigma = np.zeros((C, d, d))
    for c in range(gamma.shape[1]):
        num, den = 0, 0
        for i in range(gamma.shape[0]):
            num += gamma[i, c]*np.matmul((x[i] - mu[c])[:, np.newaxis], (x[i] - mu[c])[:, np.newaxis].transpose())
            den += gamma[i, c]
        sigma[c] = num / den

    return pi, mu, sigma


def m_step_optimized(x, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    x: (N x d), data points
    gamma: (N x C), distribution q(T)

    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = x.shape[0]
    pi = gamma.sum(axis=0) / N

    mu = np.einsum('nc, nd -> cd', gamma, x) / gamma.sum(axis=0)[:, np.newaxis]

    matrix_term = np.matmul(
        (x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
        (x[:, np.newaxis, :] - mu)[:, :, np.newaxis, :]  # transpose on last 2 terms, i.e. transpose(0, 1, 3, 2)
    )
    sigma = np.einsum(
        'nc, ncab -> ncab',
        gamma,
        matrix_term
    ).sum(axis=0) / gamma.sum(axis=0)[:, np.newaxis, np.newaxis]
    return pi, mu, sigma


def compute_vlb(x, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    x: (N x d), data points
    gamma: (N x C), distribution q(T)
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)

    Returns value of variational lower bound
    """
    N = x.shape[0]
    C = gamma.shape[1]
    d = x.shape[1]

    loss = 0
    for i in range(N):
        for c in range(C):
            norm_coeff = (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma[c])))
            gaussian_term = -0.5 * np.matmul(np.transpose(x[i] - mu[c]), np.linalg.solve(sigma[c], x[i] - mu[c]))
            loss += gamma[i, c] * (np.log(pi[c]) + np.log(norm_coeff) + gaussian_term - np.log(gamma[i, c]))
            # for numerical stability, the np.log(np.exp(gaussian)) has been simply written as gaussian

    return loss


def compute_vlb_optimized(x, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    x: (N x d), data points
    gamma: (N x C), distribution q(T)
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)

    Returns value of variational lower bound
    """

    d = x.shape[1]

    norm_coeff = (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma)))
    gaussian_terms = - 0.5 * np.einsum(
        'ijkl, ijkl -> ij',
        (x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
        np.linalg.solve(sigma, (x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis])
    )
    loss = (gamma * (np.log(pi) + np.log(norm_coeff) + gaussian_terms - np.log(gamma))).sum()
    # for numerical stability, the np.log(np.exp(gaussian)) has been simply written as gaussian

    return loss


def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.

    X: (N, d), data points
    C: int, number of clusters
    '''
    N = X.shape[0]  # number of objects
    d = X.shape[1]  # dimension of each object
    best_loss = None
    best_pi = None
    best_mu = None
    best_sigma = None

    losses, pis, mus, sigmas = [], [], [], []
    for _ in range(restarts):
        try:
            pi = np.random.rand(C)
            pi = pi / pi.sum()  # normalisation
            mu = np.random.rand(C, d)
            sigma = np.repeat(np.eye(d)[np.newaxis, :, :], repeats=C, axis=0)
            losses = []
            for iter_ in range(max_iter):
                # print(pi)
                gamma = e_step_optimized(X, pi, mu, sigma)
                pi, mu, sigma = m_step_optimized(X, gamma)
                loss = compute_vlb_optimized(X, pi, mu, sigma, gamma)
                losses.append(loss)
                if iter_ > 0 and loss < losses[iter_ - 1]:
                    raise ValueError("The vlb loss is increasing, there is a bug somewhere!")
                if iter_ > 0 and np.abs((loss - losses[iter_ - 1]) / losses[iter_ - 1]) <= rtol:
                    losses.append(loss)
                    pis.append(pi)
                    mus.append(mu)
                    sigmas.append(sigma)
                    print(f"Reached convergence in {iter_} iterations out ot {max_iter}")
                    break
            losses.append(loss)
            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")
            pass

    best_restart_index = np.argmin(losses)
    best_loss = losses[best_restart_index]
    best_pi = pis[best_restart_index]
    best_mu = mus[best_restart_index]
    best_sigma = sigmas[best_restart_index]

    # return losses, pis, mus, sigmas

    return best_loss, best_pi, best_mu, best_sigma


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
    np.identity
