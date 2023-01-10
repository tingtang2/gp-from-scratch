import argparse
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from numpy.random import Generator, default_rng
from scipy.spatial.distance import cdist

from typing import Tuple


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

    args = parser.parse_args()
    configs = args.__dict__

    rng = default_rng(configs['seed'])

    if configs['viz_draws']:
        draw_brownian_motion(rng)

    if configs['viz_prior_draws']:
        draw_prior(rng)

    return 0


if __name__ == '__main__':
    sys.exit(main())