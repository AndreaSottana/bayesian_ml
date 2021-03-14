import GPy
import GPyOpt
import numpy as np
import logging
import sklearn
from sklearn.svm import SVR
import xgboost
from matplotlib import pyplot as plt
from typing import Callable, Optional, Any


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


class BayesianOptimizer:
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


class HyperparametersFinder:
    def __init__(
            self,
            x_input: np.ndarray,
            y_output: np.ndarray,
            model_type: str,
            bounds: list[dict[str, Any]],
            acquisition_type: str = 'MPI',
            f: Optional[Callable] = None,
            baseline: Optional[float] = None
    ):
        assert any([model_type == type_ for type_ in ('xgboost', 'svr', 'custom')]), \
            f"model_type must be one of ('xgboost', 'svr', 'custom'). Got {model_type} instead."
        self.x = x_input
        self.y = y_output
        self.bounds = bounds
        if model_type == 'xgboost':
            if f is not None:
                logger.warning("Providing f when model_type == 'xgboost' is of no effect")
            if baseline is not None:
                logger.warning("Providing baseline when model_type == 'xgboost' is of no effect")
            self.f = self._f_xgboost
            self.baseline = - sklearn.model_selection.cross_val_score(
                xgboost.XGBRegressor(), self.x, self.y, scoring='neg_mean_squared_error'
            ).mean()
        elif model_type == 'svr':
            if f is not None:
                logger.warning("Providing f when model_type == 'svr' is of no effect")
            if baseline is not None:
                logger.warning("Providing baseline when model_type == 'svr' is of no effect")
            self.f = self._f_svr
            self.baseline = - sklearn.model_selection.cross_val_score(
                SVR(gamma='auto'), self.x, self.y, scoring='neg_mean_squared_error'
            ).mean()
        if model_type == 'custom':
            if f is None or baseline is None:
                raise ValueError("f and baseline must be provided manually when model_type == 'custom'.")
            self.f = f
            self.baseline = baseline
            
        self.optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.f, domain=self.bounds, acquisition_type=acquisition_type, acquisition_par=0.1, exact_feval=True
        )
        self.model_optimized = False

    def _f_xgboost(self, parameters):
        parameters = parameters[0]
        score = - sklearn.model_selection.cross_val_score(
            xgboost.XGBRegressor(
                learning_rate=parameters[0],
                max_depth=int(parameters[2]),  # convert to int in case it was picked up as float
                n_estimators=int(parameters[3]),
                gamma=int(parameters[1]),
                min_child_weight=parameters[4]
            ),
            self.x,
            self.y,
            scoring='neg_mean_squared_error'
        ).mean()
        score = np.array(score)
        return score

    def _f_svr(self, parameters):
        parameters = parameters[0]
        score = - sklearn.model_selection.cross_val_score(
            sklearn.svm.SVR(C=parameters[0], epsilon=parameters[1], gamma=parameters[2]),
            self.x,
            self.y,
            scoring='neg_mean_squared_error'
        ).mean()
        score = np.array(score)
        return score

    def optimize(self, max_iter: int, max_time: int):
        """
        Runs Bayesian Optimization inplace for a number 'max_iter' of iterations (after the initial exploration data)
        :param max_iter: exploration horizon, or maximum number of acquisitions.
        :param max_time: maximum exploration horizon in seconds.
        :return:
        """
        self.optimizer.run_optimization(max_iter, max_time, verbosity=False)
        self.model_optimized = True

    @property
    def best_parameters(self) -> dict[str, float]:
        if not self.model_optimized:
            raise ModelNotTrainedError("The model has not yet been trained.")
        best_parameters = {
            param['name']: value for param, value in zip(self.bounds, self.optimizer.X[np.argmin(self.optimizer.Y)])
        }
        return best_parameters

    @property
    def performance_boost(self) -> dict[str, float]:
        if not self.model_optimized:
            raise ModelNotTrainedError("The model has not yet been trained.")
        return self.baseline / np.min(self.optimizer.Y)

    def plot_convergence(self, show_plot: bool = False) -> plt:
        """
        Makes two plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param show_plot: whether to display the plot. Default: False.
        :return: plt: the updated matplotlib.pyplot status
        """
        if not self.model_optimized:
            raise ModelNotTrainedError("The model has not yet been trained, cannot plot convergence.")
        self.optimizer.plot_convergence()
        if show_plot:
            plt.show()
        return plt


class ModelNotTrainedError(Exception):
    pass


