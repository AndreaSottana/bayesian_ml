import numpy as np
from matplotlib import pyplot as plt
from src.baysian_opt import generate_points, generate_noise, BayesianOptimization


if __name__ == '__main__':
    x_points, y_points = generate_points(func=np.sin)
    x_points_no_noise, y_points_no_noise = generate_points(func=np.sin, noise_variance=0.0)
    x_noise_only, y_noise_only = generate_noise(noise_variance=5.0)
    plt.plot(x_points, y_points, '.')
    plt.show()
    plt.plot(x_points_no_noise, y_points_no_noise, '.')
    plt.show()
    plt.plot(x_noise_only, y_noise_only, '.')
    plt.show()

    # Try signal with noise
    opt = BayesianOptimization(x_points, y_points)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)
    print(opt.model)
    print(opt.kernel)
    print(opt.kernel.lengthscale)

    # Try signal without noise
    opt = BayesianOptimization(x_points_no_noise, y_points_no_noise)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

    # Try noise only
    opt = BayesianOptimization(x_noise_only, y_noise_only)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

    # Try signal with noise and using only 10 inducing inputs instead of whole dataset
    x_points, y_points = generate_points(func=np.sin, n=1000)
    opt = BayesianOptimization(x_points, y_points, inducing_inputs=10)
    opt.plot_model(show_plot=True)
    opt.optimize()
    opt.plot_model(show_plot=True)

