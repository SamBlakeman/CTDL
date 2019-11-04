from Gym.Parameters import env_params, agent_params
from Gym.Functions.Run import Run

def RunTask():

    for i in range(env_params['num_repeats']):
        Run(env_params, agent_params)

    return
