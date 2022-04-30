import argparse
import os

from deep_hvac import logger
from deep_hvac.ppo import train_ppo
from deep_hvac.runner import make_default_env, get_results

import numpy as np
import matplotlib.pyplot as plt


def train_ppo_agent(env_name, max_steps=100000, policy_lr=3e-4, value_lr=1e-3,
                    gae_lambda=0.95, rew_discount=0.99, max_decay_steps=1e6,
                    seed=0):
    return train_ppo(
        env_name=env_name, max_steps=max_steps,
        policy_lr=policy_lr, value_lr=value_lr, gae_lambda=gae_lambda,
        rew_discount=rew_discount, max_decay_steps=max_decay_steps, seed=seed)


def run(seed, season, continuous_action, max_steps, make_plots, construction):
    logger.log("Making env...")
    if os.path.exists(f'data/results-expert-{construction}.pickle'):
        expert_performance = f'data/results-expert-{construction}.pickle'
    else:
        expert_performance = None
    env, env_name = make_default_env(
        expert_performance=expert_performance,
        discrete_action=not continuous_action, season=season,
        capacitance=construction)

    logger.log("Starting PPO training")
    ppo_agent, save_dir = train_ppo_agent(
        max_steps=max_steps, policy_lr=1e-3, value_lr=1e-4,
        env_name=env_name, max_decay_steps=1e8, seed=seed)
    logger.log("Finished PPO training")
    logger.log(f"PPO agent saved to {save_dir}")

    if make_plots:
        ppo_results = get_results(ppo_agent, env, time=7 * 30 * 24)
        times = ppo_results['timestamp'][0]
        t_int = np.array(ppo_results['t_inside']).mean(axis=0)
        t_outside = np.array(ppo_results['t_outside']).mean(axis=0)
        t_cool_stpt = np.array(ppo_results['set_cooling']).mean(axis=0)
        t_heat_stpt = np.array(ppo_results['set_heating']).mean(axis=0)
        t_bulk = np.array(ppo_results['t_bulk']).mean(axis=0)

        plt.plot(times, t_int, label='t_inside')
        plt.plot(times, t_outside, label='t_outside')
        plt.plot(times, t_bulk, label='t_bulk')
        plt.plot(times, t_cool_stpt, linestyle='dotted', alpha=0.8,
                 label='cooling setpoint')
        plt.plot(times, t_heat_stpt, linestyle='dotted', alpha=0.8,
                 label='heating setpoint')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO.')
    parser.add_argument('--seed', type=int, default=0,
                        help='PPO training seed')
    parser.add_argument('--continuous', action='store_true',
                        help='Use continuous action space.')
    parser.add_argument('--max-steps', type=int, default=1e5,
                        help='Maximum number of steps for PPO training.')
    parser.add_argument('--plot', action='store_true',
                        help='Show plots at end of agent behavior.')
    parser.add_argument('--season', default=None, choices=('summer', 'winter'),
                        help='Season to train on')
    parser.add_argument('--construction', default='medium',
                        choices=('very_light', 'light', 'medium', 'heavy',
                                 'very_heavy'),
                        help='Building capacitance construction')

    args = parser.parse_args()

    run(args.seed, args.season, args.continuous, args.max_steps, args.plot,
        args.construction)
