import numpy as np
import pandas as pd
import pymc3 as pm
import logging
import arviz as az
import warnings
from matplotlib import pyplot as plt
from typing import Union, Optional


logger = logging.getLogger(__name__)


class BayesianLogisticRegression:
    """
    Class to perform Bayesian inference with a logistic regression model
    """
    def __init__(self, data: Union[pd.DataFrame, dict[str, np.ndarray]], *, output: str, inputs: list[str]) -> None:
        """
        :param data: a pd.DataFrame, or dictionary, containing the data to be used. If a dictionary, all values must
               be lists of the same length.
        :param output: the name of the output to be calculated
        :param inputs: the values contributing to the formula. A linear formula will be used, so if you need to sqaure
               a value you should square it here. E.g. for a formula 'income ~ sex + age**2 + educ + hours', you
               should set 'income' as your output and ['sex', 'age**2', 'educ', 'hours'] as your input.
        """
        self.data = data
        if isinstance(data, dict):
            assert output in data.keys(), f"{output} not in data"
        elif isinstance(data, pd.DataFrame):
            assert output in data.columns, f"{output} not in data"
        else:
            raise TypeError(f"data must be a dict or pd.DataFrame, got {type(data)} instead.")
        self.formula = f"{output} ~ {' + '.join(inputs)}"
        logger.warning(f"Using formula: '{self.formula}'")
        self.map_estimate = None  # placeholder for local maximum a posteriori point given a model
        self.trace = None

    def find_map_and_sample(
            self,
            family: Union[str, pm.glm.families.Family] = pm.glm.families.Binomial(),
            priors: Optional[dict] = None,
            draws: int = 400,
            init: str = 'map',  # set to 'auto' for best results
            return_inferencedata: bool = True
    ) -> tuple[dict[str, np.ndarray], pm.backends.base.MultiTrace]:
        """
        Computes the maximum a posteriori estimate and successively and performs NUTS sampling.
        :param family:
               If a string, can be one of:
               - normal
               - student
               - binomial
               - poisson
               - negative_binomial
        :param priors: the priors to use, if applicable
        :param draws: how many samples should the NUTS sampler return. Default: 400
        :param init: the initialisation, eg. 'map' or 'auto', passed onto pymc3.sample. Default: 'map.
        :param return_inferencedata: Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True)
               object or a `MultiTrace` (False). Default: True
        :return: self.map_estimate: updates, and returns the maximum a posteriori (MAP) estimate for each parameter in
                 self.inputs, and also including the 'Intercept'. A dictionary where eahc value is a scalar np.ndarray
                 self.trace: updates, and returns the pymc3.backends.base.MultiTraceraw object containing the samples
                 from the MAP posterior distribution previously calculated, using the NUTS step methods.
        """
        with pm.Model():
            pm.glm.GLM.from_formula(
                formula=self.formula,
                data=self.data,
                priors=priors,
                family=family
            )

            self.map_estimate = pm.find_MAP()
            self.trace = pm.sample(
                draws=draws, init=init, start=None, return_inferencedata=return_inferencedata
            )
            return self.map_estimate, self.trace

    def get_confidence_interval(
            self, variable: str, min_percentile: float, max_percentile: float, burnin: int = 200
    ) -> Optional[tuple[float, float]]:
        """
        Calculates the confidence interval of a given variable, within the minimum and maximum percentile given. For
        example, if you want to know the possible value of a variable with 95% confidence, you would set
        min_percentile == 2.5 and max_percentile == 97.5. The function will then return the lower and upper bounds of
        the values for the given variable, so you can have 95% confidence that the value of the variable falls within
        the returned upper and lower bounds.
        :param variable: the variable whose confidence interval is to be calculated
        :param min_percentile: the minimum percentile
        :param max_percentile:the maximum percentile
        :param burnin: the number of initial steps to discard from the trace, i.e. the points samples from the
               distribution. This is so to enable the samples to be representatives of the distributions to be
               approximated and prevent the random starting point from spoiling the data too much.
        :return: lower_bound: the lowest possible value of the variable consistent with your given min_percentile
                 upper_bound: the highest possible value of the variable consistent with your given max_percentile
        """
        if self.trace is None:
            logger.warning(
                "trace has not yet been created. Call find_map_and_sample before attempting to calculate "
                "the confidence interval"
            )
            return None
        else:
            assert min_percentile < max_percentile, 'min_percentile must be smaller than max_percentile'
            assert 0 <= min_percentile <= 100, f'min_percentile must be between 0 and 100, got {min_percentile} instead'
            assert 0 <= max_percentile <= 100, f'max_percentile must be between 0 and 100, got {max_percentile} instead'
            b = self.trace[variable][burnin:]
            lower_bound, upper_bound = np.percentile(b, min_percentile), np.percentile(b, max_percentile)
            return lower_bound, upper_bound

    def plot_traces(self, burnin: int = 200, show_plot: bool = False) -> Optional[plt]:
        """
        Convenience function to plot the traces with overlaid means and values.
        :param burnin: the number of initial steps to discard. This is so to enable the samples to be representatives
               of the distributions to be approximated and prevent the random starting point from spoiling the data
               too much.
        :param show_plot: whether to display the plot. Default: False.
        :return: plt: the updated matplotlib.pyplot status
        """
        if self.trace is None:
            logger.warning(
                "trace has not yet been created. Call find_map_and_sample before attempting to plot the trace"
            )
            return None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)  #  disables a range of warnings
                ax = az.plot_trace(
                    self.trace[burnin:],
                    figsize=(12, len(self.trace.varnames) * 1.5),
                    lines={k: v['mean'] for k, v in az.summary(self.trace[burnin:]).iterrows()}
                )

                for i, mn in enumerate(az.summary(self.trace[burnin:])['mean']):
                    ax[i, 0].annotate(
                        '{:.2f}'.format(mn), xy=(mn, 0), xycoords='data', xytext=(5, 10), textcoords='offset points',
                        rotation=90, va='bottom', fontsize='large', color='#AA0022'
                    )
                if show_plot:
                    plt.show()
                return plt

    def plot_odds_ratio_hist(
            self, variable: str, burnin: int = 200, bins: int = 20, show_plot: bool = False
    ) -> Optional[plt]:
        """
        Convenience function to plot the histogram of the odds ratio for any given variable.
        :param variable: the variable to be plotted
        :param burnin: the number of initial steps to discard. This is so to enable the samples to be representatives
               of the distributions to be approximated and prevent the random starting point from spoiling the data
               too much.
        :param bins: the number of bins of the histogram
        :param show_plot: whether to display the plot. Default: False.
        :return: plt: the updated matplotlib.pyplot status
        """
        if self.trace is None:
            logger.warning(
                "trace has not yet been created. Call find_map_and_sample before attempting to plot the trace"
            )
            return None
        else:
            b = self.trace[variable][burnin:]
            plt.hist(np.exp(b), bins=bins)
            plt.xlabel("Odds Ratio")
            plt.title(f'Variable: {variable}')
            if show_plot:
                plt.show()
            return plt
