import argparse
import os

from deep_hvac import util

import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('env_name',
                        help='Environment name.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Training seed.')
    args = parser.parse_args()
    path = os.path.join('data', args.env_name, f'seed_{args.seed}')
    steps, returns, success_rate = util.read_tf_log(path)
    fig, ax = plt.subplots()
    ax.plot(steps, returns)
    ax.set_ylabel('returns')
    ax.set_xlabel('steps')
    plt.show()
