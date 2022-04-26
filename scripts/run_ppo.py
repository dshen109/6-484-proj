import argparse

from deep_hvac import logger
from deep_hvac.ppo import train_ppo
from deep_hvac.runner import make_default_env, get_results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_ppo_agent(env_name, max_steps=100000, policy_lr=3e-4, value_lr=1e-3,
                   gae_lambda=0.95, rew_discount=0.99, seed=0):
    return train_ppo(
        env_name=env_name, max_steps=max_steps,
        policy_lr=policy_lr, value_lr=value_lr, gae_lambda=gae_lambda,
        rew_discount=rew_discount, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--seed', type=int, default=0,
                        help='PPO training seed')
    parser.add_argument('--discrete', action='store_true',
                        help='Use discrete action space.')
    parser.add_argument('--max-steps', type=int,
                        help='Maximum number of steps for PPO training.')

    args = parser.parse_args()

    logger.log("Making env...")
    env, env_name = make_default_env(
        expert_performance='data/results-expert.pickle',
        discrete_action=args.discrete)
    logger.log("Starting PPO training")
    ppo_agent, save_dir = make_ppo_agent(
        max_steps=args.max_steps, policy_lr=1e-3, value_lr=1e-2,
        env_name=env_name, seed=args.seed)
    logger.log("Finished PPO training")
    logger.log(f"Results saved to {save_dir}")

    ppo_results = get_results(ppo_agent, env, time=0)

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
