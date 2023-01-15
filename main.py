import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy.random import Generator, default_rng
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def draw_brownian_motion(rng: Generator,
                         mean: float = 0.0,
                         total_time: float = 1.0,
                         num_steps: int = 75,
                         num_draws: int = 5) -> None:

    delta_time = total_time / num_steps

    std_dev = np.sqrt(delta_time)

    # simulate gaussian random walk for brownian motion
    displacements = np.cumsum(rng.normal(mean,
                                         std_dev,
                                         size=(num_draws, num_steps)),
                              axis=1)

    data_dict = {
        'time': np.tile(np.arange(0, total_time, delta_time), reps=num_draws),
        'displacement': displacements.reshape(-1),
        'draw': [i for i in range(num_draws) for j in range(num_steps)]
    }
    df = pd.DataFrame(data_dict)

    fig = px.line(df, x='time', y='displacement', color='draw')
    fig.show()


# define kernel functions
def rbf_kernel(x_0, x_1, sigma: int = 1.0) -> float:
    squared_norm = -1 / (2 * sigma**2) * cdist(x_0, x_1, metric='sqeuclidean')
    return np.exp(squared_norm)


def draw_prior(rng: Generator,
               num_samples: int = 41,
               num_draws: int = 5,
               domain: Tuple[int, int] = (-4, 4)):

    # draw domain samples
    X_samples = np.expand_dims(np.linspace(*domain, num=num_samples), axis=1)
    covars = rbf_kernel(X_samples, X_samples)

    # draw function samples from prior
    f_samples = rng.multivariate_normal(mean=np.zeros(num_samples),
                                        cov=covars,
                                        size=num_draws)

    data_dict = {
        'x': np.tile(X_samples.reshape(-1), reps=num_draws),
        'y = f(x)': f_samples.reshape(-1),
        'draw':
        [i for i in range(num_draws) for j in range(X_samples.shape[0])]
    }
    df = pd.DataFrame(data_dict)

    fig = px.line(df, x='x', y='y = f(x)', color='draw')
    fig.show()


def infer_GP_posterior(X_train, y_train, X_test,
                       kernel) -> Tuple[np.ndarray, np.ndarray]:
    covar_train_train = kernel(X_train, X_train)
    covar_train_test = kernel(X_train, X_test)

    covar_weights = solve(covar_train_train, covar_train_test,
                          assume_a='pos').T

    # inferred y means are weighted averaged of observed covariances and observed ys
    conditional_test_mu = covar_weights @ y_train

    covar_test_test = kernel(X_test, X_test)
    conditional_test_covar = covar_test_test - (
        covar_weights @ covar_train_test)

    return conditional_test_mu, conditional_test_covar


def viz_posterior(rng: Generator,
                  num_train_points: int = 8,
                  num_test_points: int = 75,
                  domain: Tuple[int, int] = (-6, 6),
                  num_draws: int = 5):
    # true function
    f_cos = lambda x: np.cos(x).flatten()

    # generate training samples
    X_train = rng.uniform(domain[0] + 2,
                          domain[1] - 2,
                          size=(num_train_points, 1))
    y_train = f_cos(X_train)

    # uniformly predict on posterior
    X_test = np.linspace(*domain, num=num_test_points).reshape(-1, 1)

    mu_test, covar_test = infer_GP_posterior(X_train, y_train, X_test,
                                             rbf_kernel)

    sigma_test = np.sqrt(np.diag(covar_test))

    # viz fit
    fig, ax = plt.subplots()
    ax.plot(X_test, f_cos(X_test), 'b--', label='$cos(x)$')
    ax.fill_between(x=X_test.flat,
                    y1=mu_test - 2 * sigma_test,
                    y2=mu_test + 2 * sigma_test,
                    color='red',
                    alpha=0.15,
                    label='$2 \sigma_{test|train}$')
    ax.plot(X_test, mu_test, 'r-', lw=2, label='$\mu_{test|train}$')
    ax.plot(X_train, y_train, 'ko', linewidth=2, label='Training samples')
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title('Distribution of posterior and prior data.')
    ax.legend()
    plt.show()

    # viz posterior draws
    f_samples = rng.multivariate_normal(mean=mu_test,
                                        cov=covar_test,
                                        size=num_draws)

    data_dict = {
        'x': np.tile(X_test.reshape(-1), reps=num_draws),
        'y': f_samples.reshape(-1),
        'draw': [i for i in range(num_draws) for j in range(X_test.shape[0])]
    }
    df = pd.DataFrame(data_dict)

    fig = px.line(df, x='x', y='y', color='draw')
    fig.show()


def create_posterior_animation(rng: Generator,
                               total_num_train_points: int = 10,
                               num_test_points: int = 75,
                               domain: Tuple[int, int] = (-8, 8)):
    # true function
    f_cos = lambda x: np.cos(x).flatten()

    fig, ax = plt.subplots()

    # uniformly predict on posterior
    X_test = np.linspace(*domain, num=num_test_points).reshape(-1, 1)
    training_points.append(rng.uniform(domain[0] + 2, domain[1] - 2))
    X_train = np.array(training_points).reshape(-1, 1)
    y_train = f_cos(X_train)
    mu_test, covar_test = infer_GP_posterior(X_train, y_train, X_test,
                                             rbf_kernel)

    sigma_test = np.sqrt(np.diag(covar_test))
    # viz fit
    base, = ax.plot(X_test, f_cos(X_test), 'b--', label='$cos(x)$')
    region, = ax.fill_between(x=X_test.flat,
                              y1=mu_test - 2 * sigma_test,
                              y2=mu_test + 2 * sigma_test,
                              color='red',
                              alpha=0.15,
                              label='$2 \sigma_{test|train}$')
    line, = ax.plot(X_test, mu_test, 'r-', lw=2, label='$\mu_{test|train}$')
    samples, = ax.plot(X_train,
                       y_train,
                       'ko',
                       linewidth=2,
                       label='Training samples')
    ax.legend()
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title('Distribution of posterior and prior data.')

    training_points = []

    def animate(i):

        return region, line, samples

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  interval=20,
                                  blit=True,
                                  save_count=50)
    plt.show()


def infer_GP_posterior_noisy(X_train, y_train, X_test, kernel,
                             noise: float) -> Tuple[np.ndarray, np.ndarray]:
    covar_train_train = kernel(X_train, X_train) + (
        (noise**2) * np.eye(X_train.shape[0]))
    covar_train_test = kernel(X_train, X_test)

    covar_weights = solve(covar_train_train, covar_train_test,
                          assume_a='pos').T

    # inferred y means are weighted averaged of observed covariances and observed ys
    conditional_test_mu = covar_weights @ y_train

    covar_test_test = kernel(X_test, X_test)
    conditional_test_covar = covar_test_test - (
        covar_weights @ covar_train_test)

    return conditional_test_mu, conditional_test_covar


def viz_posterior_noisy(rng: Generator,
                        num_train_points: int = 8,
                        num_test_points: int = 75,
                        domain: Tuple[int, int] = (-6, 6),
                        num_draws: int = 5,
                        noise: float = .1):
    # true function
    f_cos = lambda x: np.cos(x).flatten()

    # generate training samples
    X_train = rng.uniform(domain[0] + 2,
                          domain[1] - 2,
                          size=(num_train_points, 1))
    y_train = f_cos(X_train) + (
        (noise**2) + rng.standard_normal(num_train_points))

    # uniformly predict on posterior
    X_test = np.linspace(*domain, num=num_test_points).reshape(-1, 1)

    mu_test, covar_test = infer_GP_posterior_noisy(X_train,
                                                   y_train,
                                                   X_test,
                                                   rbf_kernel,
                                                   noise=noise)

    sigma_test = np.sqrt(np.diag(covar_test))

    # viz fit
    fig, ax = plt.subplots()
    ax.plot(X_test, f_cos(X_test), 'b--', label='$cos(x)$')
    ax.fill_between(x=X_test.flat,
                    y1=mu_test - 2 * sigma_test,
                    y2=mu_test + 2 * sigma_test,
                    color='red',
                    alpha=0.15,
                    label='$2 \sigma_{test|train}$')
    ax.plot(X_test, mu_test, 'r-', lw=2, label='$\mu_{test|train}$')
    ax.plot(X_train, y_train, 'ko', linewidth=2, label='Training samples')
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$y$', fontsize=13)
    ax.set_title('Distribution of posterior and prior data.')
    ax.legend()
    plt.show()

    # viz posterior draws
    f_samples = rng.multivariate_normal(mean=mu_test,
                                        cov=covar_test,
                                        size=num_draws)

    data_dict = {
        'x': np.tile(X_test.reshape(-1), reps=num_draws),
        'y': f_samples.reshape(-1),
        'draw': [i for i in range(num_draws) for j in range(X_test.shape[0])]
    }
    df = pd.DataFrame(data_dict)

    fig = px.line(df, x='x', y='y', color='draw')
    fig.show()


def main() -> int:
    parser = argparse.ArgumentParser(description='running gp stuff')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy')
    parser.add_argument('--viz_draws',
                        action='store_true',
                        help='viz brownian motion draws')
    parser.add_argument('--viz_prior_draws',
                        action='store_true',
                        help='viz draw from prior')
    parser.add_argument('--viz_posterior',
                        action='store_true',
                        help='viz posterior with uncertainties')
    parser.add_argument('--animate_posterior_update',
                        action='store_true',
                        help='show animation of posterior updates')
    parser.add_argument('--viz_posterior_noisy',
                        action='store_true',
                        help='viz noisy posterior with uncertainties')

    args = parser.parse_args()
    configs = args.__dict__

    rng = default_rng(configs['seed'])

    if configs['viz_draws']:
        draw_brownian_motion(rng)

    if configs['viz_prior_draws']:
        draw_prior(rng)

    if configs['viz_posterior']:
        viz_posterior(rng)

    if configs['animate_posterior_update']:
        create_posterior_animation(rng)

    if configs['viz_posterior_noisy']:
        viz_posterior_noisy(rng)

    return 0


if __name__ == '__main__':
    sys.exit(main())