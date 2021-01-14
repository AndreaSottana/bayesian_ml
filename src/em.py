import numpy as np


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
                 belonging to cluster c={0, 1, ..., C-2, C-1}
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
        Each input is a numpy array:
        x: (N x d), data points
        gamma: (N x C), distribution q(T)

        :param gamma: (N x C), the posterior distribution over the latent variables q(T). N is the number of data points
               and C the number of latent variables (i.e. for GMM, number of Gaussians, or number of clusters)
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
        Computes the value of the variational lower bound, which represents our loss function.
        ### DOCSTRING TO BE FINISHED
        Each input is a numpy array:
        x: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)

        Returns value of variational lower bound
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

    def fit(self, C, rtol=1e-3, max_iter=100, restarts=10, mu_search_space=(0.0, 1.0)):
        """
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        x: (N, d), data points
        C: int, number of clusters
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
                        print(f"Reached convergence in {iter_} iterations out ot {max_iter}")
                        break
                    loss = current_loss
                if best_loss is None or loss > best_loss:
                    best_loss = loss
                    self.pi = pi
                    self.mu = mu
                    self.sigma = sigma
                    self.gamma = gamma

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass
        return best_loss, self.pi, self.mu, self.sigma

    def transform(self):
        try:
            return self._e_step(self.pi, self.mu, self.sigma)
        except TypeError:
            raise Exception("EM has not been trained, you must train with .fit before applying .transform")

    def fit_transform(self, C, rtol=1e-3, max_iter=100, restarts=10, mu_search_space=(0.0, 1.0)):
        self.fit(C, rtol=rtol, max_iter=max_iter, restarts=restarts, mu_search_space=mu_search_space)
        return self._e_step(self.pi, self.mu, self.sigma)
