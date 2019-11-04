import numpy as np
import matplotlib.pyplot as plt


def PlotComparisons(data_frames, labels):

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cumulative Episode Reward')

    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    colors = ['b', 'r', 'g', 'k', 'c', 'm']

    for df, label, color in zip(data_frames, labels, colors):

        reward_results = []
        cum_reward_results = []

        for rewards, lengths in zip(df['rewards'], df['lengths']):
            reward_results.append(rewards)
            cum_reward_results.append(np.cumsum(rewards))

        y = np.mean(reward_results, axis=0)
        x = np.arange(y.shape[0])
        error = np.std(reward_results, axis=0)

        axes[0].plot(x, y, color=color, label=label)
        axes[0].fill_between(x, y-error, y+error, color=color, alpha=.25)

        y = np.mean(cum_reward_results, axis=0)
        x = np.arange(y.shape[0])
        error = np.std(cum_reward_results, axis=0)

        axes[1].plot(x, y, color=color, label=label)
        axes[1].fill_between(x, y - error, y + error, color=color, alpha=.25)

    for s in axes.ravel():
        s.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig('Plots/ComparisonPlot.pdf')
    plt.close(fig)

    return