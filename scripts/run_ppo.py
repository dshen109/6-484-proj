import random

from deep_hvac import logger
from deep_hvac.ppo import train_ppo
from deep_hvac.runner import make_default_env, get_results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_ppo_agent(max_steps=100000):
    random.seed(12)
    return train_ppo(env_name='DefaultBuilding-v0', max_steps=max_steps)


if __name__ == "__main__":
    env = make_default_env(expert_performance='data/results-expert.pickle')
    ppo_agent, save_dir = make_ppo_agent(max_steps=1e4)

    logger.log("Starting PPO training")
    ppo_results = get_results(ppo_agent, env, time=0)
    logger.log("Finished PPO training")

    pd.to_pickle(ppo_results, 'ppo_results.pickle')

    times = ppo_results['timestamp'][0]
    t_int = np.array(ppo_results['t_inside']).mean(axis=0)
    t_outside = np.array(ppo_results['t_outside']).mean(axis=0)
    t_cool_stpt = np.array(ppo_results['set_cooling']).mean(axis=0)
    t_heat_stpt = np.array(ppo_results['set_heating']).mean(axis=0)

    plt.plot(times, t_int, label='t_inside')
    plt.plot(times, t_outside, label='t_outside')
    plt.plot(times, t_cool_stpt, linestyle='dotted', label='cooling setpoint')
    plt.plot(times, t_heat_stpt, linestyle='dotted', label='heating setpoint')
    plt.legend()
    plt.show()
