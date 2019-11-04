from GridWorld.Parameters import maze_params, agent_params
from GridWorld.Functions.Run import Run
from GridWorld.Classes.Maze import MazeType

def RunMazeTypeSweep():

    maze_types = [MazeType.direct, MazeType.obstacle1, MazeType.obstacle2]

    for i in range(maze_params['num_repeats']):
        for maze_type in maze_types:
            maze_params['type'] = maze_type
            Run(maze_params, agent_params)

    return
