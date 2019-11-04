from GridWorld.Enums.Enums import MazeType, AgentType


maze_params = {'type': MazeType.random,
               'width': 10,
               'height': 10,
               'num_rewards': 1,
               'num_trials': 1000,
               'random_seed': 0,
               'max_steps': 1000,
               'num_repeats': 30
               }

agent_params = {'agent_type': AgentType.CTDL,
                'bSOM': True,
                'SOM_alpha': .01,
                'SOM_sigma': .1,
                'SOM_sigma_const': .1,
                'Q_alpha': .9,
                'w_decay': 10,
                'TD_decay': 1,
                'SOM_size': 6,
                'e_trials': 200
                }