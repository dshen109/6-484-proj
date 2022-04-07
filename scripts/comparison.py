import os, sys
from collections import defaultdict
import numpy as np

from plot_tf_log import plot_curves

hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)
from naive import naive_agent, make_default_env, update_results
from run_ppo import make_ppo_agent

def run_episode(agent, env, episode_steps):
    env.reset()
    for _ in range(episode_steps):
        action = agent.get_action(env.get_obs())
        env.step(action[0])
    return env.results

def get_results(agent, env, episode_steps=1024, max_steps=100000):

    results = defaultdict(list)
    total_steps=0

    while total_steps < max_steps:
        ep_results = run_episode(agent, env, episode_steps)
        update_results(results, ep_results)
        total_steps += episode_steps

    return results

if __name__ == "__main__":

    # make environment of Houston
    print("Making environment")
    env = make_default_env()

    # run steps and collect results on naive_agent
    print("Running naive agent")
    naive_results = naive_agent()

    # run steps and collect results on ppo_agent
    print("Running PPO agent")
    agent_ppo, _ =  make_ppo_agent(max_steps = 100000)
    ppo_results = get_results(agent_ppo, env)

    price_dict = {'PPO Price':[[i for i in range(1024)], list(np.mean(np.array(ppo_results['price']), axis=0))],
                  'Naive Price': [[i for i in range(1024)], list(np.mean(np.array(naive_results['price']), axis=0))]}

    air_dict = {'PPO Air':[[i for i in range(1024)], list(np.mean(np.array(ppo_results['t_air']), axis=0))],
                  'Naive Air': [[i for i in range(1024)], list(np.mean(np.array(naive_results['t_air']), axis=0))]}

    settings_dict = {'PPO Heating':[[i for i in range(1024)], list(np.mean(np.array(ppo_results['set_heating']), axis=0))],
                  'Naive Heating': [[i for i in range(1024)], list(np.mean(np.array(naive_results['set_heating']), axis=0))],
                  'PPO Cooling':[[i for i in range(1024)], list(np.mean(np.array(ppo_results['set_cooling']), axis=0))],
                  'Naive Cooling': [[i for i in range(1024)], list(np.mean(np.array(naive_results['set_cooling']), axis=0))]}


    plot_curves(price_dict, "Average Price Per Step for 100 Random Episodes", xlabel='Step', ylabel='Price')
    plot_curves(air_dict, "Average Air Temperature Per Step for 100 Random Episodes", xlabel='Step', ylabel='Temp')
    plot_curves(settings_dict, "Average Thermostat Settings Per Step for 100 Random Episodes", xlabel='Step', ylabel='Temp')