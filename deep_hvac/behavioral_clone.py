from pathlib import Path

from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.envs.dummy_vec_env import DummyVecEnv
from easyrl.runner.nstep_runner import EpisodicRunner
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from deep_hvac import logger


class TrajDataset(Dataset):
    def __init__(self, trajs):
        states = []
        actions = []
        for traj in trajs:
            states.append(traj.obs)
            actions.append(traj.actions)
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        sample['state'] = self.states[idx]
        sample['action'] = self.actions[idx]
        return sample

    def add_traj(self, traj=None, states=None, actions=None):
        if traj is not None:
            self.states = np.concatenate((self.states, traj.obs), axis=0)
            self.actions = np.concatenate((self.actions, traj.actions), axis=0)
        else:
            self.states = np.concatenate((self.states, states), axis=0)
            self.actions = np.concatenate((self.actions, actions), axis=0)


def train_bc_agent(agent, trajs, max_epochs=5000, batch_size=256, lr=0.0005,
                   agent_discrete=True):
    dataset = TrajDataset(trajs)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    optimizer = optim.Adam(agent.actor.parameters(),
                           lr=lr)
    if isinstance(agent.env, DummyVecEnv):
        env = agent.env.envs[0]
    else:
        env = agent.env

    logs = dict(loss=[], epoch=[])
    for iter in range(max_epochs):
        avg_loss = []
        for batch_idx, sample in enumerate(dataloader):
            states = sample['state'].float().to(cfg.alg.device)
            expert_actions = sample['action'].float().to(cfg.alg.device)
            # Convert continuous expert actions to discrete case
            if agent_discrete:
                expert_actions = torch.tensor(
                    env.continuous_action_to_discrete(
                        expert_actions[:, 0, 0], expert_actions[:, 0, 1]
                    )
                )
            optimizer.zero_grad()
            action_dist, _ = agent.actor(states)
            loss = - (action_dist.log_prob(expert_actions)).mean()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        logs['loss'].append(np.mean(avg_loss))
        logs['epoch'].append(iter)
        if iter and iter % 100 == 0:
            logger.debug(f"Iteration {iter} of {max_epochs}")
    return agent, logs, len(dataset)


def generate_demonstration_data(expert_agent, env, num_trials):
    return run_inference(expert_agent, env, num_trials, return_on_done=True)


def run_inference(
        agent, env, num_trials, return_on_done=False, sample=True):
    runner = EpisodicRunner(agent=agent, env=env)
    trajs = []
    for _ in range(num_trials):
        env.reset()
        traj = runner(time_steps=cfg.alg.episode_steps,
                      sample=sample,
                      return_on_done=return_on_done,
                      evaluation=True,
                      render_image=False)
        trajs.append(traj)
    return trajs


def set_configs(env_name, exp_name='bc', seed=0):
    set_config('ppo')
    cfg.alg.seed = seed
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 24 * 30
    cfg.alg.max_steps = 600000
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = env_name
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += f'/{exp_name}'
    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))
