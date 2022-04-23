from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def main():
    data_folder = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )

    steps, returns, success_rate = read_tf_log(os.path.join(data_folder, 'DefaultBuilding-v0'))

    plot_curves({'reward': [steps, returns]}, 'PPO Rewards')


def plot_curves(data_dict, title, xlabel='Steps', ylabel='Reward'):
    # {label: [x, y]}
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = data_dict.keys()
    for label, data in data_dict.items():
        x = data[0]
        y = data[1]
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()