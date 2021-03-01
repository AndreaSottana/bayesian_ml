import numpy as np
import pandas as pd
import pymc3 as pm
import logging
from typing import Union, Optional


logger = logging.getLogger(__name__)


class BayesianLogisticRegressionManual:
    def __init__(self, data: Union[pd.DataFrame, dict[str, np.ndarray]], *, output: str, inputs: list[str]):
        self.data = data
        if isinstance(data, dict):
            assert output in data.keys(), f"{output} not in data"
            assert all([input_ in data.keys() for input_ in inputs]), f"some of {inputs} are not in data"
        elif isinstance(data, pd.DataFrame):
            assert output in data.columns, f"{output} not in data"
            assert all([input_ in data.columns for input_ in inputs]), f"some of {inputs} are not in data"
        else:
            raise TypeError(f"data must be a dict or pd.DataFrame, got {type(data)} instead.")
        self.output = output
        self.inputs = inputs
        self.map_estimate = None  # placeholder for local maximum a posteriori point given a model

    def find_map_manual(
            self,
            mus: list[float],
            sigmas: list[float],
            family: Union[str, pm.glm.families.Family] = pm.glm.families.Binomial(),
            priors: Optional[dict] = None
    ):
        """

        :param mus: intercept u first
        :param sigmas: intercept sigma first
        :param family:
        :param priors:
        :return:
        """
        with pm.Model():
            parameters = {'a': pm.Normal('a', mu=mus.pop(0), sigma=sigmas.pop(0))}
            parameters.update({
                f'b{i + 1}': pm.Normal(f'b{i + 1}', mu=mu, sigma=sigma)
                for i, (mu, sigma) in enumerate(zip(mus, sigmas))
            })

            sum_ = np.ones(len(self.data))
            for (param, param_value), input_ in zip(parameters.items(), self.inputs):
                if param == 'a':
                    sum_ *= param_value
                else:
                    sum_ += self.data[input_].values * param_value

            logits = pm.invlogit(sum_)

            pm.Bernoulli('Bernoulli', p=logits, observed=self.data[self.output].values)

            self.map_estimate = pm.find_MAP()
            return self.map_estimate


