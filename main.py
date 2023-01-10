import argparse
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from numpy.random import Generator, default_rng


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


def main() -> int:
    parser = argparse.ArgumentParser(description='running gp stuff')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy')
    parser.add_argument('--viz_draws',
                        action='store_true',
                        help='viz brownian motion draws')
    args = parser.parse_args()
    configs = args.__dict__

    rng = default_rng(configs['seed'])

    if configs['viz_draws']:
        draw_brownian_motion(rng)
    return 0


if __name__ == '__main__':
    sys.exit(main())