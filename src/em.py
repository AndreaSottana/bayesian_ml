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
    :return: gamma: (N x C), probabilities of clusters for objects; each component along axis = 0 represents
             a point, each component along axis = 1 represent the probability of that point belonging to cluster
             c=[0, C].
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
    """
    assert pi.shape[0] == mu.shape[0] == sigma.shape[0], "Some shapes are incompatible"
    assert x.shape[1] == mu.shape[1] == sigma.shape[1] == sigma.shape[2], "Some shapes are incompatible"
    N = x.shape[0]  # number of objects
    C = pi.shape[0]  # number of clusters

    gaussians = np.zeros((N, C))
    for i in range(N):
        gaussians[i] = np.einsum('ij,ij->i', (x[i] - mu), np.linalg.solve(sigma, x[i] - mu))
    gaussians = gaussians - np.max(gaussians, axis=1)[:, np.newaxis]  # trick for numerical stability
    gaussians = np.exp(-0.5 * gaussians)
    weighted_gaussians = pi * gaussians / np.sqrt(np.linalg.det(sigma))
    gamma = weighted_gaussians / np.sum(weighted_gaussians, axis=1)[:, np.newaxis]
    return gamma


if __name__ == '__main__':
    samples = np.load('../data/samples.npz')
    X_ = samples['data']
    pi0_ = samples['pi0']
    mu0_ = samples['mu0']
    sigma0_ = samples['sigma0']
    print(e_step(X_, pi0_, mu0_, sigma0_)[:7])
    print(e_step_optimized(X_, pi0_, mu0_, sigma0_)[:7])


