import pandas as pd
from src.mcmc import BayesianLogisticRegression


if __name__ == '__main__':
    df = pd.read_csv('../data/adult_us_postprocessed.csv')
    lr = BayesianLogisticRegression(df, output='income_more_50K', inputs=['age', 'educ'])
    map_, trace = lr.find_map_and_sample(draws=2)
    print(map_, trace)
    lr.plot_traces()
    lb, up = lr.get_confidence_interval('sex[T. Male]', min_percentile=2.5, max_percentile=97.5)
    print(f"We are 95% confident that the correlation between gender and income is between {lb} and {up}.")
    lr.plot_traces(show_plot=True)
    lr.plot_odds_ratio_hist('sex[T. Male]', show_plot=True)
