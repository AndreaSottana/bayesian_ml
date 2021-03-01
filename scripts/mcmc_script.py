import pandas as pd
from src.mcmc import BayesianLogisticRegression
from src.mcmc_manual import BayesianLogisticRegressionManual


if __name__ == '__main__':
    df = pd.read_csv('../data/adult_us_postprocessed.csv')
    lr = BayesianLogisticRegression(df, output='income_more_50K', inputs=['age', 'educ'])
    map, trace = lr.find_map_and_sample(draws=2)
    print(map, trace)
    lr.plot_traces()
    lr.plot_odds_ratio_hist('sex[T. Male]')

    lr = BayesianLogisticRegressionManual(df, output='income_more_50K', inputs=['age', 'educ'])
    print(lr.find_map_manual(mus=[0, 0, 0], sigmas=[100, 100, 100]))
