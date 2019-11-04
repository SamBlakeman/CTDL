import numpy as np
import matplotlib.pyplot as plt


def PlotComparisons(var, data_frames, labels):

    vals = np.array([])
    for df in data_frames:
        vals = np.concatenate([vals, df[var].values])

    vals = np.unique(vals)
    num_plots = vals.shape[0]

    figs = []
    axes = []
    for i in range(num_plots):
        f, a = plt.subplots(4, figsize=(3, 6))

        a[0].axis('off')
        a[3].set_xlabel('Episode')
        a[1].set_ylabel('Episode Length')
        a[2].set_ylabel('Reward')
        a[3].set_ylabel('Ideal Episodes')
        a[1].set_xticks([])
        a[2].set_xticks([])

        a[3].spines['top'].set_visible(False)
        a[3].spines['right'].set_visible(False)
        a[1].spines['top'].set_visible(False)
        a[1].spines['right'].set_visible(False)
        a[1].spines['bottom'].set_visible(False)
        a[2].spines['top'].set_visible(False)
        a[2].spines['right'].set_visible(False)
        a[2].spines['bottom'].set_visible(False)

        a[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        a[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        figs.append(f)
        axes.append(a)

    colors = ['b', 'r', 'g', 'k', 'c', 'm']


    for df, label, color in zip(data_frames, labels, colors):

        length_results = [[] for i in range(num_plots)]
        reward_results = [[] for i in range(num_plots)]
        ideal_results = [[] for i in range(num_plots)]

        for v, rewards, lengths, maze in zip(df[var], df['rewards'], df['lengths'], df['maze']):

            p = np.where(vals == v)[0][0]

            axes[p][0].set_title(var + ': ' + str(v))
            axes[p][0].imshow(maze)

            length_results[p].append(np.cumsum(lengths))
            reward_results[p].append(np.cumsum(rewards))
            ideal_results[p].append(np.cumsum(np.array(rewards) == 1))


        for p in range(num_plots):

            if(length_results[p]):

                y = np.mean(length_results[p], axis=0)
                x = np.arange(y.shape[0])
                error = np.std(length_results[p], axis=0)

                axes[p][1].plot(x, y, color=color)
                axes[p][1].fill_between(x, y-error, y+error, color=color, alpha=.25)

                y = np.mean(reward_results[p], axis=0)
                x = np.arange(y.shape[0])
                error = np.std(reward_results[p], axis=0)

                axes[p][2].plot(x, y, color=color)
                axes[p][2].fill_between(x, y - error, y + error, color=color, alpha=.25)

                y = np.mean(ideal_results[p], axis=0)
                x = np.arange(y.shape[0])
                error = np.std(ideal_results[p], axis=0)

                axes[p][3].plot(x, y, label=label, color=color)
                axes[p][3].fill_between(x, y - error, y + error, color=color, alpha=.25)

    for a in axes:
        for s in a.ravel():
            s.legend()

    for i, f in enumerate(figs):
        f.tight_layout()
        f.savefig('Plots/ComparisonPlot' + str(i) + '.pdf')
        plt.close(f)

    return


def PlotPairwiseComparison(df1, df2, labels):

    vals = np.array([])
    vals = np.concatenate([vals, df1['random_seed'].values])
    vals = np.concatenate([vals, df2['random_seed'].values])

    vals = np.unique(vals)
    num_points = vals.shape[0]

    reward_results = [[] for i in range(num_points)]
    ideal_results = [[] for i in range(num_points)]

    for seed, rewards, lengths in zip(df1['random_seed'], df1['rewards'], df1['lengths']):
        p = np.where(vals == seed)[0][0]
        reward_results[p].append(np.sum(rewards))
        ideal_results[p].append(np.sum(np.array(rewards) == 1))

    ys = np.zeros((2, num_points))

    for p in range(num_points):
        ys[0, p] = np.mean(reward_results[p])
        ys[1, p] = np.mean(ideal_results[p])


    reward_results = [[] for i in range(num_points)]
    ideal_results = [[] for i in range(num_points)]

    for seed, rewards, lengths in zip(df2['random_seed'], df2['rewards'], df2['lengths']):
        p = np.where(vals == seed)[0][0]
        reward_results[p].append(np.sum(rewards))
        ideal_results[p].append(np.sum(np.array(rewards) == 1))

    xs = np.zeros((2, num_points))

    for p in range(num_points):
        xs[0, p] = np.mean(reward_results[p])
        xs[1, p] = np.mean(ideal_results[p])


    colors = ['r', 'b']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    axes[0].scatter(xs[0, :], ys[0, :], color=[colors[i] for i in ys[0, :] > xs[0, :]])
    axes[1].scatter(xs[1, :], ys[1, :], color=[colors[i] for i in ys[1, :] > xs[1, :]])

    min_val = np.min(np.concatenate([xs[0, :], ys[0, :]]))
    max_val = np.max(np.concatenate([xs[0, :], ys[0, :]]))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k-')
    axes[0].axis('equal')
    axes[0].set_aspect('equal', 'box')

    min_val = np.min(np.concatenate([xs[1, :], ys[1, :]]))
    max_val = np.max(np.concatenate([xs[1, :], ys[1, :]]))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k-')
    axes[1].axis('equal')
    axes[1].set_aspect('equal', 'box')

    axes[0].set_ylabel(labels[0])
    axes[0].set_xlabel(labels[1])
    axes[1].set_xlabel(labels[1])

    axes[0].set_title('Reward')
    axes[1].set_title('Ideal Episodes')

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig('Plots/PairwiseComparisonPlot.pdf')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].pie([np.sum(ys[0, :] > xs[0, :]), np.sum(ys[0, :] < xs[0, :])], colors=reversed(colors))
    axes[1].pie([np.sum(ys[1, :] > xs[1, :]), np.sum(ys[1, :] < xs[1, :])], colors=reversed(colors))
    fig.tight_layout()
    plt.savefig('Plots/PairwisePieChart.pdf')
    plt.close()

    return


def PlotMeanSOMLocations(root_dir, df):

    vals = df['type'].values
    vals = np.unique(vals)
    num_plots = vals.shape[0]

    mazes = [[] for i in range(num_plots)]

    for type, directory in zip(df['type'], df['dir']):
        som_locations = np.load(root_dir + directory + '/SOMLocations.npy')

        p = np.where(vals == type)[0][0]
        mazes[p].append(som_locations)

    for i in range(num_plots):
        plt.figure()
        plt.imshow(np.mean(mazes[i], axis=0))#, cmap='plasma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('Plots/MeanSOMLocations' + str(i) + '.pdf')
        plt.close()



    return


def PlotRevaluationComparisons(data_frames, labels):

    start = 0
    end = 1000

    var = 'type'
    vals = np.array([])
    for df in data_frames:
        vals = np.concatenate([vals, df[var].values])

    vals = np.unique(vals)
    num_mazes = vals.shape[0]

    f, a = plt.subplots(3, figsize=(6, 6))

    a[2].set_xlabel('Episode')
    a[0].set_ylabel('Episode Length')
    a[1].set_ylabel('Reward')
    a[2].set_ylabel('Ideal Episodes')
    a[0].set_xticks([])
    a[1].set_xticks([])

    a[2].spines['top'].set_visible(False)
    a[2].spines['right'].set_visible(False)
    a[0].spines['top'].set_visible(False)
    a[0].spines['right'].set_visible(False)
    a[0].spines['bottom'].set_visible(False)
    a[1].spines['top'].set_visible(False)
    a[1].spines['right'].set_visible(False)
    a[1].spines['bottom'].set_visible(False)

    a[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    a[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    colors = ['b', 'r', 'g', 'k', 'c', 'm']

    for df, label, color in zip(data_frames, labels, colors):

        length_results = [[] for i in range(num_mazes)]
        reward_results = [[] for i in range(num_mazes)]
        ideal_results = [[] for i in range(num_mazes)]

        for v, rewards, lengths, maze in zip(df[var], df['rewards'], df['lengths'], df['maze']):
            p = np.where(vals == v)[0][0]

            length_results[p].append(np.cumsum(lengths))
            reward_results[p].append(np.cumsum(rewards))
            ideal_results[p].append(np.cumsum(np.array(rewards) == 1))

        for p in range(num_mazes):

            num_trials = df['num_trials'][0]

            if(p != 0):
                y = np.array(length_results[p]) + np.expand_dims(np.array(length_results[p - 1])[:, -1], axis=-1)
            else:
                y = length_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            a[0].plot(x, y, color=color)
            a[0].fill_between(x, y - error, y + error, color=color, alpha=.25)

            if (p != 0):
                y = np.array(reward_results[p]) + np.expand_dims(np.array(reward_results[p - 1])[:, -1], axis=-1)
            else:
                y = reward_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            a[1].plot(x, y, color=color)
            a[1].fill_between(x, y - error, y + error, color=color, alpha=.25)

            if (p != 0):
                y = np.array(ideal_results[p]) + np.expand_dims(np.array(ideal_results[p-1])[:, -1], axis=-1)
            else:
                y = ideal_results[p]

            error = np.std(y, axis=0)
            y = np.mean(y, axis=0)
            x = np.arange(y.shape[0]) + (p * num_trials)

            if(p==0):
                a[2].plot(x, y, label=label, color=color)
            else:
                a[2].plot(x, y, color=color)
                a[0].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)
                a[1].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)
                a[2].axvline(p * num_trials, color='k', linestyle='--', linewidth=2)

            a[2].fill_between(x, y - error, y + error, color=color, alpha=.25)

    for axis in a:
        axis.set_xlim([start, end])

    for s in a.ravel():
        s.legend()

    f.tight_layout()
    f.savefig('Plots/RevaluationComparisonPlot.pdf')
    plt.close(f)

    return