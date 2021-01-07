import numpy as np


def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices

    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0]  # number of objects
    C = pi.shape[0]  # number of clusters
    d = X.shape[1]  # dimension of each object, also equal to mu.shape[1]
    gamma = np.zeros((N, C))  # distribution q(T)

    ### YOUR CODE HERE
    y_max = np.max(pi * (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma))))
    for i in range(N):
        weighted_gaussians = np.zeros(C)
        for c in range(C):
            norm_coeff = (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma0[c])))
            gaussian = norm_coeff * np.exp(
                -0.5 * np.matmul(np.transpose(X[i] - mu[c]), np.linalg.solve(sigma[c], X[i] - mu[c])))
            weighted_gaussian = pi[c] * gaussian
            weighted_gaussians[c] = weighted_gaussian
        gamma[i] = weighted_gaussians / np.sum(weighted_gaussians)

    return gamma



def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices

    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0]  # number of objects
    C = pi.shape[0]  # number of clusters
    d = X.shape[1]  # dimension of each object, also equal to mu.shape[1]
    gamma = np.zeros((N, C))  # distribution q(T)

    ### YOUR CODE HERE
    y_max = np.max(pi * (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma))))
    for i in range(N):
        gaussians = np.zeros(C)
        for c in range(C):
            gaussians[c] = (
                np.exp(-0.5 * np.matmul(np.transpose(X[i] - mu[c]), np.linalg.solve(sigma[c], X[i] - mu[c])))
            )
        weighted_gaussians = pi * gaussians / np.sqrt(np.linalg.det(sigma))
        gamma[i] = weighted_gaussians / np.sum(weighted_gaussians)

    return gamma


