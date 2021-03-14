import GPy
import GPyOpt
import numpy as np
from typing import Callable, Optional


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

