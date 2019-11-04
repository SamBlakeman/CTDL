import os
import gym

from datetime import datetime
from Utilities import RecordSettings
from Gym.Enums.Enums import EnvType, AgentType


def Run(env_params, agent_params):

    results_dir = CreateResultsDirectory()

    # Setup envrionment
    if (env_params['env'] == EnvType.CartPole):
        env = gym.make('CartPole-v1')
    elif (env_params['env'] == EnvType.MountainCarContinuous):
        env = gym.make('MountainCarContinuous-v0')

    env_params['env_obj'] = env
    env_params['state_mins'] = env.observation_space.low
    env_params['state_maxs'] = env.observation_space.high
    env_params['state_dim'] = env.observation_space.shape[0]

    if(isinstance(env.action_space, gym.spaces.Box)):
        env_params['action_maxs'] = env.action_space.high
        env_params['action_mins'] = env.action_space.low
    else:
        env_params['num_actions'] = env.action_space.n

    # Setup agent
    if(agent_params['agent_type'] == AgentType.CTDL):
        from Gym.Agents.CTDL.Agent import Agent
    elif(agent_params['agent_type'] == AgentType.DQN):
        from Gym.Agents.DQN.Agent import Agent
    elif (agent_params['agent_type'] == AgentType.A2C):
        from Gym.Agents.A2C.Agent import Agent
    elif (agent_params['agent_type'] == AgentType.CTDL_A2C):
        from Gym.Agents.CTDL_A2C.Agent import Agent

    agent = Agent(results_dir, env_params, agent_params)

    # Record settings
    RecordSettings(results_dir, env_params, agent_params)

    # Run
    RunEnv(agent, env, env_params)

    return


def RunEnv(agent, env, env_params):

    trial = 0
    reward = 0
    bTrial_over = False
    state = env.reset()
    ti = 0

    print('Starting Trial ' + str(trial) + '...')
    while trial < env_params['num_trials']:

        if (ti % 50 == 0):
            print('Time Step: ' + str(ti) + ' Agent Epsilon: ' + str(agent.epsilon))
        ti += 1

        action = agent.Update(reward, state, bTrial_over)
        state, reward, bTrial_over, info = env.step(action)

        if(ti % env_params['max_steps'] == 0):
            bTrial_over = True

        if (bTrial_over):
            trial += 1
            ti = 0
            state = env.reset()
            print('Starting Trial ' + str(trial) + '...')

    env.close()
    agent.PlotResults()


def CreateResultsDirectory():
    date_time = str(datetime.now())
    date_time = date_time.replace(" ", "_")
    date_time = date_time.replace(".", "_")
    date_time = date_time.replace("-", "_")
    date_time = date_time.replace(":", "_")
    # Make the results directory
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    results_dir = dir_path + '/../Results/' + date_time + '/'
    os.mkdir(results_dir)
    return results_dir



