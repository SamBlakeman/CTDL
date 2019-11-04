import os
from datetime import datetime

from Utilities import RecordSettings
from GridWorld.Classes.Maze import Maze
from GridWorld.Enums.Enums import AgentType

def Run(maze_params, agent_params):

    results_dir = CreateResultsDirectory()
    maze_params['num_hazards'] = int((maze_params['width'] * maze_params['height']) / 5)
    RecordSettings(results_dir, maze_params, agent_params)

    if(agent_params['agent_type'] == AgentType.CTDL):
        from GridWorld.Agents.CTDL.Agent import Agent
    elif(agent_params['agent_type'] == AgentType.DQN):
        from GridWorld.Agents.DQN.Agent import Agent

    agent = Agent(results_dir, maze_params, agent_params)
    maze = Maze(results_dir, maze_params)

    RunMaze(agent, maze, maze_params)

    return


def RunSequentially(maze_params, agent_params, mazes):

    maze_params['num_hazards'] = int((maze_params['width'] * maze_params['height']) / 5)
    agent_params['e_trials'] = 200#int(maze_params['num_trials'] / 5)

    if (agent_params['agent_type'] == AgentType.CTDL):
        from GridWorld.Agents.CTDL.Agent import Agent
    elif (agent_params['agent_type'] == AgentType.DQN):
        from GridWorld.Agents.DQN.Agent import Agent

    for i, m in enumerate(mazes):

        maze_params['type'] = m
        results_dir = CreateResultsDirectory()

        if(i == 0):
            agent = Agent(results_dir, maze_params, agent_params)
        else:
            agent.NewMaze(results_dir)

        RecordSettings(results_dir, maze_params, agent_params)

        maze = Maze(results_dir, maze_params)
        RunMaze(agent, maze, maze_params)

    return


def RunMaze(agent, maze, maze_params):
    trial = 0
    reward = 0
    state = maze.start
    bTrial_over = False
    ti = 0
    print('Starting Trial ' + str(trial) + '...')
    while trial < maze_params['num_trials']:

        if (ti % 50 == 0):
            print('Time Step: ' + str(ti) + ' Agent Epsilon: ' + str(agent.epsilon))
        ti += 1

        action = agent.Update(reward, state, bTrial_over)
        reward, state, bTrial_over = maze.Update(action)

        if (bTrial_over):
            trial += 1
            ti = 0
            print('Starting Trial ' + str(trial) + '...')

            if (trial % 10 == 0):
                agent.PlotValueFunction()
    agent.PlotResults()
    return


def CreateResultsDirectory():
    date_time = str(datetime.now())
    date_time = date_time.replace(" ", "_")
    date_time = date_time.replace(".", "_")
    date_time = date_time.replace("-", "_")
    date_time = date_time.replace(":", "_")
    # Make the results directory
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = dir_path + '/Results/' + date_time + '/'
    os.mkdir(results_dir)
    return results_dir



