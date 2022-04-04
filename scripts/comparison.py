import os, sys

from plot_tf_log import plot_curves

hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)
from naive import naive_agent, make_default_env
from run_ppo import make_ppo_agent

def get_results(agent, env):
    action = agent.model(env.get_obs())
    print(action)
    pass

if __name__ == "__main__":

    # make environment of Houston
    env = make_default_env()

    # run steps and collect results on naive_agent
    # naive_results = naive_agent()

    # run steps and collect results on ppo_agent
    agent_ppo, _ =  make_ppo_agent()
    ppo_results = get_results(agent_ppo, env)