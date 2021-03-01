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
        :param output: tje name of the output to be calculated
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
    ):
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
        :return:
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

    def plot_traces(self, burnin: int = 200, show_plot: bool = False):
        """
        Convenience function to plot the traces with overlaid means and values.
        :param burnin: the number of initial steps to discard. This is so to enable the samples to be representatives
               of the distributions to be approximated and prevent the random starting point from spoiling the data
               too much.
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
                return ax

    def plot_odds_ratio_hist(self, variable, burnin: int = 200, bins: int = 20, show_plot: bool = False):
        if self.trace is None:
            logger.warning(
                "trace has not yet been created. Call find_map_and_sample before attempting to plot the trace"
            )
            return None
        else:
            b = self.trace[variable][burnin:]
            plt.hist(np.exp(b), bins=bins)
            plt.xlabel("Odds Ratio")
            plt.title(variable)
            if show_plot:
                plt.show()
            return plt
