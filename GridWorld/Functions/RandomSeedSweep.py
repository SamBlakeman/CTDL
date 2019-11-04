import numpy as np

from GridWorld.Parameters import maze_params, agent_params
from GridWorld.Functions.Run import Run

def RunRandomSeedSweep():

    random_seeds = np.arange(0, 50).tolist()

    for i in range(maze_params['num_repeats']):
        for random_seed in random_seeds:
            maze_params['random_seed'] = random_seed
            Run(maze_params, agent_params)

    return
