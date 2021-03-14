import GPy
import GPyOpt
import numpy as np
import logging
from matplotlib import pyplot as plt
from typing import Callable, Optional


logger = logging.getLogger(__name__)


def generate_points(
        func: Callable,
        n: int = 25,
        noise_variance: float = 0.0036,
        low: float = -3.0,
        high: float = 3.0,
        size: Optional[tuple[int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates sampled points for a given function (eg. np.sin(x)) with added noise
    :param func: the callable function to generate points from. E.g. for y=np.sin(x), set func == np.sin
    :param n: the number of points to be generated. Default: 25
    :param noise_variance: the variance of the noise to be added to the generated points. Default: 0.0036
    :param low: the minimum value of x used to generate points. Default: -3.0
    :param high: the maximum value of x used to generate points. Default: 3.0
    :param size: the shape of the returned arrays. If None, it will set it to (n, 1) (i.e. a column vector).
           Default: None
    :return: x: a np.ndarray with the generated x points, uniformly randomly generated between low and high parameters
             y: a np.ndarray with the generated y points, using the formula y = func(x) + random_noise where the
             variance of the random noise is determined by noise_variance parameter.
    """
    if size is None:
        size = (n, 1)
    x = np.random.uniform(low, high, size)
    y = func(x) + np.random.rand(*size) * noise_variance**0.5
    return x, y


def generate_noise(
        n: int = 25,
        noise_variance: float = 0.0036,
        low: float = -3.0,
        high: float = 3.0,
        size: Optional[tuple[int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random noise within a given set of values
    :param n: the number of noise points to be generated. Default: 25
    :param noise_variance: the variance of the noise. Default: 0.0036
    :param low: the minimum value of x used to generate points. Default: -3.0
    :param high: the maximum value of x used to generate points. Default: 3.0
    :param size: the shape of the returned arrays. If None, it will set it to (n, 1) (i.e. a column vector).
           Default: None
    :return: x: a np.ndarray with the generated x points, uniformly randomly generated between low and high parameters
             y: a np.ndarray with the generated y points, i.e. the random noise whose variance is determined by
             noise_variance parameter.
    """
    if size is None:
        size = (n, 1)
    x = np.random.uniform(low, high, size)
    y = np.random.rand(*size) * noise_variance**0.5
    return x, y


class BayesianOptimization:
    def __init__(
            self,
            x_input: np.ndarray,
            y_output: np.ndarray,
            kernel_input_dim: int = 1,
            kernel_variance: float = 1.5,
            kernel_lengthscale: float = 2.0,
            inducing_inputs: Optional[int] = None,
            use_gpu: bool = False,
    ):
        self.kernel = GPy.kern.RBF(kernel_input_dim, kernel_variance, kernel_lengthscale, useGPU=use_gpu)
        if inducing_inputs is None:
            self.model = GPy.models.GPRegression(x_input, y_output, kernel=self.kernel)
        else:
            self.model = GPy.models.SparseGPRegression(
                x_input, y_output, kernel=self.kernel, num_inducing=inducing_inputs
            )
            logger.warning(f"Using {inducing_inputs} inducing inputs instead of the whole dataset")
        self.model_optimized = False

    def optimize(self, max_iters: int = 1000) -> None:
        """
        Optimize the self.model inplace using log likelihood and log likelihood gradient, as well as the priors.
        :param max_iters: maximum number of function evaluations. Default: 1000
        """
        self.model.optimize(max_iters=max_iters, ipython_notebook=False)
        self.model_optimized = True

    def predict(self, x_unseen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the function(s) at the new unseen point(s) x_unseen. This includes the likelihood variance added to
        the predicted underlying function
        :param x_unseen: a column vector, or 2-dimensional array containing the inputs
        :return: mean: the posterior mean of the output(s)
                 variance: the posterior variance of the output(s)
        """
        if len(x_unseen.shape) != 2:
            raise ValueError(
                f"x_unseen must be a column vector and have 2 dimensions. Got {len(x_unseen.shape)} dimensions instead."
            )
        if not self.model_optimized:
            logger.warning("You have not optimized / trained the model yet, therefore results might be low quality.")
        mean, variance = self.model.predict(x_unseen)
        return mean, variance

    def plot_model(self, show_plot: bool = False) -> plt:
        if not self.model_optimized:
            logger.warning("You have not optimized / trained the model yet, therefore the plot might be low quality.")
        self.model.plot()
        if show_plot:
            plt.show()
        return plt
