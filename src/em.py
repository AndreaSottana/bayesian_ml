import logging
from typing import Union
import numpy as np


logger = logging.getLogger(__name__)


class EM:
    """
    Class to perform probabilistic clustering of a set of N d-dimensional point assuming such points are generated
    from a Gaussian Mixture Model. The optimization is done via the expectation-maximization algorithm, a coordinate
    descent optimization of a variational lower bound.
    """
    def __init__(self, x: np.ndarray) -> None:
        """
        :param x: (N x d), the input data points
        """
        self.x = x
        # the parameters below will be lazily initialized when first fitting a dataset with the fit method.
        self.pi = None
        self.mu = None
        self.sigma = None
        self.gamma = None

    def _e_step(self, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Performs the E-step (expectation step) on the Gaussian Mixture Model (GMM). This step estimates the posterior
        distribution gamma over the latent variables (i.e. the Gaussians) with fixed values of parameters pi, mu and
        sigma. Each input is a numpy array. Shapes are presented below, where
        - N: number of data points
        - d: number of dimensions
        - C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)

        The method is optimised for performance due to broadcasting for speed and includes other tricks to improve
        numerical stability when calculating very large / small numbers. Some common factors in the formula,
        e.g. 1 / np.sqrt(np.power(2 * np.pi, d) have also been removed as they would be cancelled out in normalization
        operations.
        :param pi: (C), mixture component weights. They must be normalized, i.e. pi.sum() == 1.
        :param mu: (C x d), mixture component means
        :param sigma: (C x d x d), mixture component covariance matrices
        :return: gamma: (N x C), probabilities of clusters for objects; each component along axis = 0 [0, N-1]
                 represents a point, each component along axis = 1 [0, C-1] represent the probability of that point
                 belonging to cluster c={0, 1, ..., C-2, C-1}. This is the the posterior distribution over the latent
                 variables.
        """

        gaussians = np.einsum(
            'ijkl, ijkl -> ij',
            (self.x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
            np.linalg.solve(sigma, (self.x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis])
        )
        gaussians = gaussians - np.max(gaussians, axis=1)[:, np.newaxis]  # trick for numerical stability
        gaussians = np.exp(-0.5 * gaussians)
        weighted_gaussians = pi * gaussians / np.sqrt(np.linalg.det(sigma))
        gamma = weighted_gaussians / np.sum(weighted_gaussians, axis=1)[:, np.newaxis]
        return gamma

    def _m_step(self, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the M-step (maximization step) on the Gaussian Mixture Model (GMM). This step maximises the expectation
        value, with respect to the posterior, of the likelihood, over the parameters pi, mu and sigma, while the
        posterior gamma is kept fixed. Usually, the logarithm is taken before optimizing the function, because it is
        easier to optimize and doesn't change the results.
        Each input is a numpy array. Shapes are presented below, where
        - N: number of data points
        - d: number of dimensions
        - C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)

        :param gamma: (N x C), probabilities of clusters for objects; each component along axis = 0 [0, N-1]
               represents a point, each component along axis = 1 [0, C-1] represent the probability of that point
               belonging to cluster c={0, 1, ..., C-2, C-1}. This is the the posterior distribution over the latent
               variables.
        :return: pi: (C), mixture component weights. They must be normalized, i.e. pi.sum() == 1.
                 mu: (C x d), mixture component means
                 sigma: (C x d x d), mixture component covariance matrices
        """
        N = self.x.shape[0]
        pi = gamma.sum(axis=0) / N

        mu = np.einsum('nc, nd -> cd', gamma, self.x) / gamma.sum(axis=0)[:, np.newaxis]

        matrix_term = np.matmul(
            (self.x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
            (self.x[:, np.newaxis, :] - mu)[:, :, np.newaxis, :]  # transpose on last 2 terms,i.e. transpose(0, 1, 3, 2)
        )
        sigma = np.einsum(
            'nc, ncab -> ncab',
            gamma,
            matrix_term
        ).sum(axis=0) / gamma.sum(axis=0)[:, np.newaxis, np.newaxis]
        return pi, mu, sigma

    def _compute_vlb(self, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gamma: np.ndarray) -> int:
        """
        Computes the value of the variational lower bound, which represents our loss function. This is the value we
        are trying to optimize and which is used to track convergence.
        Each input is a numpy array. Shapes are presented below, where
        - N: number of data points
        - d: number of dimensions
        - C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)

        :param pi: (C), mixture component weights. They must be normalized, i.e. pi.sum() == 1.
        :param mu: (C x d), mixture component means
        :param sigma: (C x d x d), mixture component covariance matrices
        :param gamma: (N x C), the posterior distribution over the latent variables q(T). N is the number of data points
               and C the number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)
        :return: loss: the value of the variational lower bound.
        """

        d = self.x.shape[1]

        norm_coeff = (1 / np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(sigma)))
        gaussian_terms = - 0.5 * np.einsum(
            'ijkl, ijkl -> ij',
            (self.x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis],
            np.linalg.solve(sigma, (self.x[:, np.newaxis, :] - mu)[:, :, :, np.newaxis])
        )
        loss = (gamma * (
            np.log(pi + 1e-20) + np.log(norm_coeff + 1e-20) + gaussian_terms - np.log(gamma + 1e-20)
        )).sum()  # for numerical stability, the np.log(np.exp(gaussian)) has been simply written as gaussian and other
        # logarithm's terms had 1e-20 added to them to prevent np.log(0).

        return loss

    def fit(
            self, C, rtol=1e-3, max_iter=100, restarts=10, mu_search_space=(0.0, 1.0)
    ) -> int:
        """
        Implements the training loop for solving the Gaussian Mixture Model using the expectation-maximization
        algorithm. The training is re-initialized with random initialization a number of times, and the values
        returning the best (i.e. lowest in absolute value) variational lower bound loss is retained. Each restart
        is run until either saturation is reached, or a maximum number of epochs is reached.
        :param C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)
        :param rtol: the tolerance. The model will continue training until saturation, i.e. until
               abs((L{i} - L{i-1}) / L{i-1}) <= rtol, or until a maximum number of iterations is reached.
               L{i} is the loss at iteration i. Default: 1e-3
        :param max_iter: The maximum number of iterations, per each restart. If this value is reached before
               convergence / saturation, the training will stop regardless. Default: 100
        :param restarts: The number of restarts with randomly re-initialised parameters. The values generated by the
               restart with the lowest absolute loss will then be retained. Default: 10
        :param mu_search_space: a tuple of floats representing the search space for each dimension / component of the
               means of the latent Gaussian curves. Syntax: (min, max). Each dimension / component of the mean of each
               latent Gaussian will be uniformly randomly sampled within this range. Default: (0.0, 1.0)
        :return: best_loss: the final value of the loss from the best restart.
        """
        d = self.x.shape[1]
        best_loss = None

        for _ in range(restarts):
            try:
                pi = np.random.uniform(low=0.0, high=1.0, size=C)
                pi = pi / pi.sum()  # normalisation
                mu = np.random.uniform(low=mu_search_space[0], high=mu_search_space[1], size=(C, d))
                sigma = np.repeat(np.eye(d)[np.newaxis, :, :], repeats=C, axis=0)
                loss = None
                for iter_ in range(max_iter):
                    gamma = self._e_step(pi, mu, sigma)
                    pi, mu, sigma = self._m_step(gamma)
                    current_loss = self._compute_vlb(pi, mu, sigma, gamma)
                    if loss is not None and current_loss < loss:
                        raise ValueError("The vlb loss is increasing, there is a bug somewhere!")
                    if iter_ > 0 and np.abs((current_loss - loss) / loss) <= rtol:
                        logger.info(f"Reached convergence in {iter_} iterations out ot {max_iter}")
                        break
                    loss = current_loss
                if best_loss is None or loss > best_loss:
                    best_loss = loss
                    self.pi = pi
                    self.mu = mu
                    self.sigma = sigma
                    self.gamma = gamma

            except np.linalg.LinAlgError:
                logger.warning("Singular matrix: components collapsed! Skipping this run.")
                pass
        return best_loss

    def transform(self) -> np.ndarray:
        """
        Perform the probabilistic predictions on the dataset. Each point is assigned a probability to it belonging
        to each of the C latent variables (i.e. Gaussians) learned during training. You need to train the model
        first by calling .fit(...) before performing predictions, or else an exception will be raised.
        :return: gamma: (N x C), the posterior distribution over the latent variables q(T). N is the number of data
                 points and C the number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters),
                 N is the number of data points, while C is the number of latent variables (i.e. for GMM, number of
                 Gaussians, or number of clusters)
        """
        try:
            return self._e_step(self.pi, self.mu, self.sigma)
        except TypeError:
            raise Exception("EM has not been trained, you must train with .fit before applying .transform")

    def fit_transform(
            self, C, rtol=1e-3, max_iter=100, restarts=10, mu_search_space=(0.0, 1.0), return_loss: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, int]]:
        """
        Combines the fit and transform methods into one single method.
        First, it implements the training loop for solving the Gaussian Mixture Model using the expectation-maximization
        algorithm. The training is re-initialized with random initialization a number of times, and the values
        returning the best (i.e. lowest in absolute value) variational lower bound loss is retained. Each restart
        is run until either saturation is reached, or a maximum number of epochs is reached. Then, the probabilistic
        predictions on the dataset are performed. Each point is assigned a probability to it belonging
        to each of the C latent variables (i.e. Gaussians) learned during training.
        :param C: number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)
        :param rtol: the tolerance. The model will continue training until saturation, i.e. until
               abs((L{i} - L{i-1}) / L{i-1}) <= rtol, or until a maximum number of iterations is reached.
               L{i} is the loss at iteration i. Default: 1e-3
        :param max_iter: The maximum number of iterations, per each restart. If this value is reached before
               convergence / saturation, the training will stop regardless. Default: 100
        :param restarts: The number of restarts with randomly re-initialised parameters. The values generated by the
               restart with the lowest absolute loss will then be retained. Default: 10
        :param mu_search_space: a tuple of floats representing the search space for each dimension / component of the
               means of the latent Gaussian curves. Syntax: (min, max). Each dimension / component of the mean of each
               latent Gaussian will be uniformly randomly sampled within this range. Default: (0.0, 1.0)
        :param return_loss: whether to return the final value of the loss from the best training restart. Default: False
        :return: gamma: (N x C), the posterior distribution over the latent variables q(T). N is the number of data
                 points and C the number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters),
                 N is the number of data points, while C is the number of latent variables (i.e. for GMM, number of
                 Gaussians, or number of clusters)
        """
        best_loss = self.fit(C, rtol=rtol, max_iter=max_iter, restarts=restarts, mu_search_space=mu_search_space)
        gamma = self._e_step(self.pi, self.mu, self.sigma)
        if return_loss:
            return gamma, best_loss
        else:
            return gamma
