from GridWorld.Parameters import maze_params, agent_params
from GridWorld.Functions.Run import RunSequentially, AgentType
from GridWorld.Classes.Maze import MazeType

def RunRevaluationSweep():

    maze_types = [MazeType.direct, MazeType.obstacle1]

    for i in range(maze_params['num_repeats']):
        RunSequentially(maze_params, agent_params, maze_types)

    return
