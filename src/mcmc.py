import numpy as np
import pandas as pd
import pymc3 as pm
import logging
from typing import Union, Optional


logger = logging.getLogger(__name__)


class BayesianLogisticRegression:
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
        self.formula = f"{output} ~ {' + '.join(inputs)}"
        logger.warning(f"Using formula: '{self.formula}'")
        self.map_estimate = None  # placeholder for local maximum a posteriori point given a model

    def find_map(
            self, family: Union[str, pm.glm.families.Family] = pm.glm.families.Binomial(), priors: Optional[dict] = None
    ):
        """

        :param family:
               If a string, can be one of:
               - normal
               - student
               - binomial
               - poisson
               - negative_binomial
        :param priors
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
            return self.map_estimate
