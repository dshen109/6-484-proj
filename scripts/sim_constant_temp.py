import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from deep_hvac.naive import naive_agent
from deep_hvac.naive import make_default_env


hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)


def plot(data, title, xlabel='steps', ylabel='reward'):
    plt.figure()
    x = data[0]
    y = data[1]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    env = make_default_env()
    naive_results = naive_agent(heating_temp=30, cooling_temp=35)

    prices = [[i for i in range(1024)], list(np.mean(np.array(naive_results['price']), axis=0))]
    air_temp = [[i for i in range(1024)], list(np.mean(np.array(naive_results['t_air']), axis=0))]
    cooling = [[i for i in range(1024)], list(np.mean(np.array(naive_results['set_cooling']), axis=0))]

    plot(prices, 'cost, naive', ylabel='cost')
    plot(air_temp, 'air, naive', ylabel='air temp')
    plot(cooling, 'cooling setpoint, naive', ylabel='cooling stpt')
