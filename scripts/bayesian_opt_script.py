import numpy as np
from matplotlib import pyplot as plt
from src.bayesian_opt import generate_points, generate_noise, GaussianProcess, BayesianOptimization
from sklearn import datasets


if __name__ == '__main__':
    x_points, y_points = generate_points(func=np.sin)
    x_points_no_noise, y_points_no_noise = generate_points(func=np.sin, noise_variance=0.0)
    x_noise_only, y_noise_only = generate_noise(noise_variance=5.0)
    fig, ax = plt.subplots(3)
    fig.tight_layout()
    ax[0].plot(x_points, y_points, '.')
    ax[0].set_title("y = sin(x) + noise")
    ax[1].plot(x_points_no_noise, y_points_no_noise, '.')
    ax[1].set_title("y = sin(x) - no noise")
    ax[2].plot(x_noise_only, y_noise_only, '.')
    ax[2].set_title("noise only")
    plt.show()

    # Try signal with noise
    opt = GaussianProcess(x_points, y_points)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)
    print(opt.model)
    print(opt.kernel)
    print(opt.kernel.lengthscale)

    # Try signal without noise
    opt = GaussianProcess(x_points_no_noise, y_points_no_noise)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

    # Try noise only
    opt = GaussianProcess(x_noise_only, y_noise_only)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

    # Try signal with noise and using only 10 inducing inputs instead of whole dataset
    x_points, y_points = generate_points(func=np.sin, n=1000)
    opt = GaussianProcess(x_points, y_points, num_inducing_inputs=10)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

    # Perform hyperparameter tuning for XGBoost model using Bayesian Optimization
    data = datasets.load_diabetes()
    x = data['data']
    y = data['target']

    bounds = [  # NOTE: define continuous variables first, then discrete!
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}
    ]
    hp_finder = BayesianOptimization(data['data'], data['target'], model_type='xgboost', bounds=bounds)
    hp_finder.optimize(max_iter=50, max_time=60)
    print("baseline MSE: ", hp_finder.baseline)
    print("Best values of hyper-parameters after optimization: ", hp_finder.best_parameters)
    print("MSE after optimization: ", np.min(hp_finder.optimizer.Y))
    print("Performance boost compared to baseline after optimization: ", hp_finder.performance_boost)
    hp_finder.plot_convergence(show_plot=True)

    # Perform hyperparameter tuning for SVR model using Bayesian Optimization
    bounds = [
        {'name': 'C', 'type': 'continuous', 'domain': (1e-5, 1000)},
        {'name': 'epsilon', 'type': 'continuous', 'domain': (1e-5, 10)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (1e-5, 10)}
    ]
    hp_finder = BayesianOptimization(data['data'], data['target'], model_type='svr', bounds=bounds)
    hp_finder.optimize(max_iter=50, max_time=60)
    print("baseline MSE: ", hp_finder.baseline)
    print("Best values of hyper-parameters after optimization: ", hp_finder.best_parameters)
    print("MSE after optimization: ", np.min(hp_finder.optimizer.Y))
    print("Performance boost compared to baseline after optimization: ", hp_finder.performance_boost)
    hp_finder.plot_convergence(show_plot=True)
