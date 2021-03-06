from collections import defaultdict
import os

from deep_hvac import agent, logger, simulator
from deep_hvac.building import default_building
from deep_hvac.util import NsrdbReader

from easyrl.utils.gym_util import make_vec_env
from gym.envs.registration import registry, register
import pandas as pd


def run_episode(agent, env, episode_steps, time=0):
    """
    :param Agent agent:
    :param Env env:
    :param int episode_steps: Length of episode
    :param int time: Starting time.
    """
    env.reset(time=time)
    for _ in range(episode_steps):
        action, _ = agent.get_action(env.get_obs())
        # agent.get_action returns an action for each environment, so just
        # take the first one
        if env.config.discrete_action:
            action = action.item()
        env.step(action)
    return env.results


def get_results(agent, env, episode_steps=30 * 24,
                max_steps=7200, time=0):
    """
    :param int time: Starting time for result plotting.
    """

    results = defaultdict(list)
    total_steps = 0

    while total_steps < max_steps:
        ep_results = run_episode(agent, env, episode_steps, time=time)
        update_results(results, ep_results)
        total_steps += episode_steps

    return results


def update_results(results, ep_results):
    """Append results"""
    for k in ep_results.keys():
        results[k].append(ep_results[k])


def make_default_env(episode_length=24 * 30, terminate_on_discomfort=True,
                     discomfort_penalty=1e4, discrete_action=True,
                     expert_performance=None):
    """
    Register a default environment

    :return Env: Registered environment
    """
    if discrete_action:
        env_name = 'DefaultBuilding-v0-action-discrete'
    else:
        env_name = 'DefaultBuilding-v0-action-continuous'
    logger.debug(f"Creating default environment {env_name}.")
    config = simulator.SimConfig(
        episode_length=episode_length,
        terminate_on_discomfort=terminate_on_discomfort,
        discomfort_penalty=discomfort_penalty,
        discrete_action=discrete_action
    )
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    logger.debug("Loading NSRDB data...")
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    logger.debug("Finished loading NSRDB data.")
    logger.debug("Loading Houston price data...")
    ercot = pd.read_pickle(os.path.join(
        datadir, 'houston-2018-prices.pickle'))
    logger.debug("Finished loading price data.")
    zone, windows, latitude, longitude = default_building()

    env_args = dict(
        prices=ercot,
        weather=nsrdb.weather_hourly,
        agent=None,
        coords=(latitude, longitude),
        zone=zone,
        windows=windows,
        config=config,
        expert_performance=None
    )
    if expert_performance is None:
        # Run the naive agent as a baseline.
        naive_agent = agent.NaiveAgent()
        env = simulator.SimEnv(**env_args)
        logger.debug("Running naive baseline...")
        obs = env.reset(0)
        for _ in range(365 * 24 - 2):
            obs, _, terminate, _ = env.step(naive_agent.get_action(obs)[0])
        costs = env.results['electricity_cost']
        # pad with zeros
        costs.insert(0, 0)
        costs.append(0)
        env_args['expert_performance'] = pd.DataFrame(
            costs, columns=['cost'], index=nsrdb.weather_hourly.index)
        env.results['expert_performance'] = env_args['expert_performance']
    elif isinstance(expert_performance, str):
        expert = pd.read_pickle(expert_performance)
        if not isinstance(expert, pd.DataFrame):
            env_args['expert_performance'] = expert['expert_performance']
        else:
            env_args['expert_performance'] = expert
    else:
        env_args['expert_performace'] = expert_performance
    if env_name in registry.env_specs:
        del registry.env_specs[env_name]
    register(
        id=env_name, entry_point='simulator:SimEnv', kwargs=env_args
    )
    return make_vec_env(env_name, 1, 0).envs[0], env_name


def make_testing_env(episode_length=24 * 30, terminate_on_discomfort=False,
                     discomfort_penalty=100):
    """Create 2019 Houston dataset as testing environment."""
    # TODO: Finish.
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    logger.debug("Loading NSRDB data...")
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2019.csv'))
    logger.debug("Finished loading NSRDB data.")
    logger.debug("Loading Houston price data...")
    ercot = pd.read_pickle(os.path.join(
        datadir, 'houston-2018-prices.pickle'))
    logger.debug("Finished loading price data.")
    # TODO: finish.

