import pathlib
import os

from deep_hvac import behavioral_clone, logger, runner
from deep_hvac.agent import BasicCategoricalAgentStateSubset, NaiveAgent
from deep_hvac.simulator import SimEnv

from easyrl.utils.gym_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import torch


def mimic(env_name, actor=None, hidden_size=126):
    env = make_vec_env(env_name, 1, 0).envs[0]
    return BasicCategoricalAgentStateSubset(
        state_indices=(
            SimEnv.state_idx['hour'],
            SimEnv.state_idx['weekday'],
            SimEnv.state_idx['occupancy_ahead_0']
        ), env=env, actor=actor, hidden_size=hidden_size
    )


if __name__ == '__main__':
    scriptdir = pathlib.Path(__file__).parent.resolve()

    plot_performace = True
    season = 'aug'
    trajectory_file = os.path.join(
        scriptdir, 'fixtures', f'expert-traj-{season}.pt'
    )
    cloned_agent_save = os.path.join(
        scriptdir, 'output', f'cloned-agent-basic-{season}.pt'
    )
    max_epochs = 200
    n_trajectories = 10

    _, env_name = runner.make_default_env(
        terminate_on_discomfort=False, create_expert=False,
        discrete_action=True, season=season
    )
    env = make_vec_env(env_name, 1, 0)
    try:
        trajectories = torch.load(trajectory_file)
        logger.log(f"Loaded expert demonstrations {trajectory_file}")
    except FileNotFoundError:
        trajectories = None

    expertagent = NaiveAgent(env=env)
    behavioral_clone.set_configs(env_name)
    if trajectories is None:
        logger.log("Creating demonstration data...")
        trajectories = behavioral_clone.generate_demonstration_data(
            expertagent, env, n_trajectories)
        torch.save(trajectories, trajectory_file)
        logger.log(f"Saved demo data to {trajectory_file}")
    else:
        trajectories = trajectories
    logger.log("Starting behavioral cloning...")
    agent_cloned, logs, _ = behavioral_clone.train_bc_agent(
        mimic(env_name), trajectories[:n_trajectories],
        max_epochs=max_epochs)
    torch.save(agent_cloned.actor, cloned_agent_save)

    if plot_performace:
        actor = torch.load(cloned_agent_save)
        agent_cloned = mimic(env_name, actor)
        logger.log("Generating agent evaluation results")
        results = runner.get_results(agent_cloned, env.envs[0], time=8*30*24,
                                     episode_steps=4*24, max_steps=4*24*5)
        logger.log("Generating expert evaluation results")
        results_expert = runner.get_results(
            expertagent, env.envs[0], time=8*30*24, episode_steps=4*24,
            max_steps=4*24*2
        )
        fields = ('t_inside', 't_outside', 'set_cooling', 'set_heating')

        results_arrays = {'timestamp': results['timestamp'][0]}
        results_arrays_expert = {'timestamp': results['timestamp'][0]}
        for k in fields:
            results_arrays[k] = np.array(results[k]).mean(axis=0)
            results_arrays_expert[k] = np.array(results_expert[k]).mean(axis=0)

        fig, axs = plt.subplots(2)

        for ax, result in zip(axs, (results_arrays, results_arrays_expert)):
            for k in fields:
                if 'set_' in k:
                    linestyle = 'dotted'
                    alpha = 0.8
                else:
                    linestyle = 'solid'
                    alpha = 0.8
                ax.plot(result['timestamp'], result[k], label=k,
                        alpha=alpha, linestyle=linestyle)
            plt.legend()

        axs[0].set_title('Cloned agent')
        axs[1].set_title('Expert agent')
        plt.show()
