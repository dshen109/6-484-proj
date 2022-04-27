import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib import rc
import random
from torch import nn
from gym.envs.registration import registry, register

from building import comfort_temperature
from runner import make_default_env



class MPC:

    def __init__(self, 
        model,
        horizon        = 25,
        es_epsilon     = 0.001,
        es_alpha       = 0.1,
        es_generations = 5,
        es_popsize     = 200,
        es_elites      = 40
    ):
        self.model = model

        self.horizon        = horizon        # planning horizon
        self.es_epsilon     = es_epsilon     # variance threshold
        self.es_alpha       = es_alpha       # new distribution rolling average coefficient
        self.es_generations = es_generations # num generations for ES optimizer
        self.es_popsize     = es_popsize     # popsize for ES optimizer
        self.es_elites      = es_elites      # num of elites from which to resample

        self.reset()

    def reset(self):
        self.sol_mean = ((self.model.u_lb + self.model.u_ub) / 2).expand(self.horizon, self.model.u_shape)
        self.sol_var = ((self.model.u_ub - self.model.u_lb) / 16).expand(self.horizon, self.model.u_shape)
        
        self.timestep = 0

    def action(self, q):
        # Remove last taken action and add 0 to end of buffer
        self.sol_mean = torch.cat([
            self.sol_mean[1:],
            torch.zeros(self.model.u_shape).reshape(1,-1)
        ])        
                
        # Generate standard diagonal normal distribution from which we sample trajectory noise        
        u_dist = torch.distributions.normal.Normal(
            loc=torch.zeros_like(self.sol_mean), 
            scale=torch.ones_like(self.sol_var)
        )
        
        var = self.sol_var
        for n in range(self.es_generations):            
            # Terminate if variance drops below threshold
            if torch.max(var) < self.es_epsilon:
                print('var below threshold! exiting')
                break

            lb_dist = self.sol_mean - self.model.u_lb
            ub_dist = self.model.u_ub - self.sol_mean
            constrained_var = torch.min(
                torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var
            )
            

            new_dist = torch.distributions.normal.Normal(
                loc=torch.zeros_like(self.sol_mean), 
                scale=torch.ones_like(constrained_var)
            )
            actions = [self.sol_mean + new_dist.sample() for _ in range(self.es_popsize)]
            actions = torch.stack(actions)
            qs = torch.tensor([q for _ in range(self.es_popsize)])
            trajs = self.model.run_batch_of_trajectories(qs, actions)[:, 1:]

            rewards = self.model.reward(trajs, actions)
            
            rewards_sort = torch.argsort(torch.sum(rewards, 1), descending=True)
            elites = actions[rewards_sort][:self.es_elites]
            elites = torch.mean(elites, 0)
            elites_var = torch.var(elites, 0)

            self.sol_mean = self.es_alpha * elites + (1 - self.es_alpha) * self.sol_mean
            self.sol_var = self.es_alpha * elites_var + (1 - self.es_alpha) * self.sol_var
        
        self.timestep += 1
        return self.sol_mean[0]

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class LearnedDynamicsModel:

    def __init__(self, seed, u_range=[10, 40], discomfort_penalty = 100):
        set_random_seed(seed)
        self.u_range = u_range

        self.discomfort_penalty = discomfort_penalty

        self.u_lb = torch.tensor([u_range[0]]).float()
        self.u_ub = torch.tensor([u_range[1]]).float()
        self.u_shape = 2

        input_size = 18

        self.model = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.Tanh()
        )

        self.optim = torch.optim.Adam(self.model.parameters())
        self.loss = nn.MSELoss()

    def step(self, q, u):
        try:
          inp = torch.stack([q[:, 0], q[:, 1], q[:, 2], 
                             q[:, 3], q[:, 4], q[:, 5],
                             q[:, 6], q[:, 7], q[:, 8],
                             q[:, 9], q[:, 10], q[:, 11],
                             q[:, 12], q[:, 13], q[:, 14],
                             q[:, 15], u[:, 0], u[:, 1] 
                            ]).T
          inp = torch.tensor(inp, dtype=torch.float32)
          q_prime = q + self.model(inp)
          return q_prime
        except:
          inp = torch.stack([q[:, 0], q[:, 1], q[:, 2], 
                             q[:, 3], q[:, 4], q[:, 5],
                             q[:, 6], q[:, 7], q[:, 8],
                             q[:, 9], q[:, 10], q[:, 11],
                             q[:, 12], q[:, 13], q[:, 14],
                             q[:, 15], u[:, 0], u[:, 1] 
                            ]).T
          inp = torch.tensor(inp, dtype=torch.float32)
          q_prime = q + self.model(inp)
          return q_prime

    # given q [n, q_shape] and u [n, t] run the trajectories
    def run_batch_of_trajectories(self, q, u):
        qs = [q]
        
        for t in range(u.shape[1]):
            qs.append(self.step(qs[-1], u[:, t]))
         
        return torch.stack(qs, dim=1)

    def train(self, q_t_traj, q_tplusone_traj, u_traj):
        batch_size = 16
        num_batches = 1024

        all_batch_idxs = np.random.randint(
            len(q_t_traj), size=(num_batches, batch_size)
        )

        for b in range(num_batches):
            batch_idxs = all_batch_idxs[b]
            q, qprime, u = torch.from_numpy(q_t_traj[batch_idxs]), \
                           torch.from_numpy(q_tplusone_traj[batch_idxs]), \
                           torch.from_numpy(u_traj[batch_idxs])

            pred = self.step(q, u)

            loss = self.loss(pred, qprime)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def reward(self, q, u):
        #TODO: calculate reward by calling env 
        electricity_cost = q[:, :, 13] * q[:, :, 11] / 1000
        reward_from_price = -electricity_cost

        discomfs = np.zeros((q.shape[0], q.shape[1]))
        q = q.detach()
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                discomfs[i, j] = self.comfort_penalty(q[i, j, 2], q[i, j, 8])

        discomfs = torch.from_numpy(discomfs)
        return (discomfs * self.discomfort_penalty) + reward_from_price
        

    def comfort_penalty(self, outdoor, t_air):
        """
        Calculate ASHRAE comfort penalty.

        Is interpolated between [0.5, 1] if >2.5 or <3.5 degrees away from
        comfort temperature and |diff - 2.5| if >3.5 degrees away from comfort
        temperature.
        """
        t_comf = comfort_temperature(outdoor)
        deviation = np.abs(t_air - t_comf)
        if deviation <= 2.5:
            return 0
        elif deviation <= 3.5:
            return np.interp(deviation, [2.5, 3.5], [0.5, 1])
        else:
            return deviation - 2.5


if __name__ == "__main__":
    dynamics_model = LearnedDynamicsModel(seed=0)
    mpc = MPC(dynamics_model)
    e = make_default_env()

    all_q, all_q_prime, all_u = None, None, None
    for epoch in range(100):
        q_traj = [e.reset()]
        u_traj = []

        r = 0
        while True:
            u_traj.append(mpc.action(q_traj[-1]).numpy())
            q, reward, done, _ = e.step(u_traj[-1])
            r += reward
            q_traj.append(q)

            if done:
                break

        q_traj = np.array(q_traj, dtype=np.float32)
        u_traj = np.array(u_traj, dtype=np.float32)

        print('[Epoch ' + str(epoch) + '] Got reward ' + str(r))
        q, q_prime, u = q_traj[:-1], q_traj[1:], u_traj

        if all_q is None:
            all_q, all_q_prime, all_u = q, q_prime, u
        else:
            all_q = np.concatenate((all_q, q))
            all_q_prime = np.concatenate((all_q_prime, q_prime))
            all_u = np.concatenate((all_u, u))

        dynamics_model.train(all_q, all_q_prime, all_u)

        if r > -20:
            break

            