from copy import deepcopy
import os
import pathlib

from deep_hvac import behavioral_clone, logger, ppo, runner
from deep_hvac.agent import BasicAgent, NaiveAgent

from easyrl.configs import cfg
from easyrl.utils.gym_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import torch


def mimic(env_name, actor=None):
    """Agent that is taught to mimic the expert."""
    agent, _ = ppo.train_ppo(
        env_name=env_name, save_dir='tmp', train=False
    )
    if actor is not None:
        return BasicAgent(actor, env=agent.env)
    else:
        return BasicAgent(agent.actor, env=agent.env)


def make_ppo_agent(env_name, max_steps, save_dir, seed=0, actor=None,
                   actor_save_path=None, force_run=False):
    if isinstance(actor, str):
        try:
            actor_path = actor
            actor = torch.load(actor)
            logger.log(f"Loaded PPO actor from {actor_path}")
        except FileNotFoundError:
            actor = None
    if actor is not None and not force_run:
        return BasicAgent(
            actor=actor, env=make_vec_env(env_name, 1, 0).envs[0]), ''

    agent_ppo, ppo_save_dir = ppo.train_ppo(
        env_name=env_name, max_steps=max_steps, seed=seed,
        actor=deepcopy(actor),
        save_dir=save_dir, max_saved_models=20
    )
    if actor_save_path:
        torch.save(agent_ppo.actor, actor_save_path)
    return agent_ppo, ppo_save_dir


def run_bc_and_ppo_pipeline(trajectory_file, cloned_agent_save, seed=0):
    max_epochs = 300
    n_trajectories = 20
    ppo_steps = 100000
    action_change_penalty = 100
    _, env_name = runner.make_default_env(
        terminate_on_discomfort=False, create_expert=False,
        discrete_action=True, season=season, capacitance=capacitance,
        action_change_penalty=action_change_penalty
    )
    _, env_name_ppo = runner.make_default_env(
        terminate_on_discomfort=True, create_expert=True,
        discrete_action=True, season=season, capacitance=capacitance
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
            expertagent, env, 30)
        torch.save(trajectories, trajectory_file)
        logger.log(f"Saved expert demonstrations to {trajectory_file}")
    else:
        trajectories = trajectories
    logger.log(f"Starting behavioral cloning for seed {seed}...")
    # agent_cloned, logs, _ = behavioral_clone.train_bc_agent(
    #     mimic(env_name), trajectories[:n_trajectories],
    #     max_epochs=max_epochs)
    cloned_agent_save_seed = \
        f'{cloned_agent_save[:-3]}-seed_{seed}.pt'
    # torch.save(agent_cloned.actor, cloned_agent_save_seed)

    logger.log("Starting PPO tuning on cloned agent...")
    agent_ppo, ppo_save_dir = make_ppo_agent(
        env_name_ppo, max_steps=ppo_steps,
        save_dir=f'bc-ppo-tune-{ppo_steps}-' + env_name_ppo,
        seed=seed, actor=cloned_agent_save_seed, actor_save_path=None,
        force_run=True)
    logger.log(f"PPO saved to {ppo_save_dir}")


if __name__ == '__main__':
    plot_performance = True
    scriptdir = pathlib.Path(__file__).parent.resolve()
    season = 'summer'
    capacitance = 'medium'
    trajectory_file = os.path.join(
        scriptdir, 'fixtures', f'expert-traj-{season}.pt'
    )
    cloned_agent_save = os.path.join(
        scriptdir, 'output', f'cloned-actor-basic-{season}.pt'
    )
    ppo_actor_save = os.path.join(
        scriptdir, 'output', f'ppo-tuned-actor-{capacitance}-{season}.pt'
    )


    for seed in (2, 1, 0):
        run_bc_and_ppo_pipeline(trajectory_file, cloned_agent_save, seed)

    # max_epochs = 500
    # n_trajectories = 20

    # _, env_name = runner.make_default_env(
    #     terminate_on_discomfort=False, create_expert=False,
    #     discrete_action=True, season=season, capacitance=capacitance
    # )
    # _, env_name_ppo = runner.make_default_env(
    #     terminate_on_discomfort=True, create_expert=True,
    #     discrete_action=True, season=season, capacitance=capacitance
    # )
    # env = make_vec_env(env_name, 1, 0)
    # try:
    #     trajectories = torch.load(trajectory_file)
    #     logger.log(f"Loaded expert demonstrations {trajectory_file}")
    # except FileNotFoundError:
    #     trajectories = None

    # expertagent = agent.NaiveAgent(env=env)
    # behavioral_clone.set_configs(env_name)

    # try:
    #     agent_cloned = mimic(env_name, actor=torch.load(cloned_agent_save))
    #     logger.log(f"Loaded cloned agent from {cloned_agent_save}")
    # except Exception:
    #     if trajectories is None:
    #         logger.log("Creating demonstration data...")
    #         trajectories = behavioral_clone.generate_demonstration_data(
    #             expertagent, env, 30)
    #         torch.save(trajectories, trajectory_file)
    #         logger.log(f"Saved expert demonstrations to {trajectory_file}")
    #     else:
    #         trajectories = trajectories
    #     logger.log("Starting behavioral cloning...")
    #     agent_cloned, logs, _ = behavioral_clone.train_bc_agent(
    #         mimic(env_name), trajectories[:n_trajectories],
    #         max_epochs=max_epochs)
    #     torch.save(agent_cloned.actor, cloned_agent_save)

    # # # Do PPO tuning on the cloned agent.
    # logger.log("Starting PPO tuning on cloned agent...")
    # agent_ppo, _ = make_ppo_agent(
    #     env_name_ppo, max_steps=1e5,
    #     save_dir='bc-ppo-tune-' + env_name_ppo,
    #     seed=0, actor=agent_cloned.actor, actor_save_path=ppo_actor_save,
    #     force_run=True)

    # if plot_performance:
    #     time = 8 * 30 * 24
    #     episode_steps = 4 * 24
    #     max_steps = episode_steps * 5

    #     actor = torch.load(cloned_agent_save)
    #     agent_cloned = agent.BasicAgent(
    #         actor=actor, env=make_vec_env(env_name, 1, 0))
    #     agent_ppo = agent.BasicAgent(
    #         actor=torch.load(ppo_actor_save), env=make_vec_env(env_name, 1, 0)
    #     )
    #     logger.log("Generating PPO agent evaluation results")
    #     outputs_ppo = runner.get_results(
    #         agent_ppo, env.envs[0], time=time, episode_steps=episode_steps,
    #         max_steps=max_steps
    #     )
    #     logger.log("Generating agent evaluation results")
    #     outputs_cloned = runner.get_results(
    #         agent_cloned, env.envs[0], time=8*30*24,
    #         episode_steps=4*24, max_steps=4*24*5)
    #     logger.log("Generating expert evaluation results")
    #     outputs_expert = runner.get_results(
    #         expertagent, env.envs[0], time=8*30*24, episode_steps=4*24,
    #         max_steps=int(max_steps / 4)
    #     )
    #     fields = ('t_inside', 't_bulk', 't_outside',
    #               'set_cooling', 'set_heating')

    #     results_cloned = {'timestamp': outputs_cloned['timestamp'][0]}
    #     results_expert = {'timestamp': outputs_cloned['timestamp'][0]}
    #     results_ppo = deepcopy(results_expert)
    #     for k in fields:
    #         results_cloned[k] = np.array(outputs_cloned[k]).mean(axis=0)
    #         results_expert[k] = np.array(outputs_expert[k]).mean(axis=0)
    #         results_ppo[k] = np.array(outputs_ppo[k]).mean(axis=0)

    #     fig, axs = plt.subplots(3)

    #     for ax, result in zip(axs,
    #                           (results_ppo, results_cloned, results_expert)):
    #         for k in fields:
    #             if 'set_' in k:
    #                 linestyle = 'dotted'
    #                 alpha = 0.8
    #             else:
    #                 linestyle = 'solid'
    #                 alpha = 0.8
    #             ax.plot(result['timestamp'], result[k], label=k,
    #                     alpha=alpha, linestyle=linestyle)
    #         plt.legend()

    #     axs[0].set_title('PPO agent')
    #     axs[1].set_title('Cloned agent')
    #     axs[2].set_title('Expert agent')
    #     plt.show()
