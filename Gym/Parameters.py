from Gym.Enums.Enums import EnvType, AgentType

env_params = {'env': EnvType.MountainCarContinuous,
              'num_trials': 200,
              'max_steps': 1000,
              'num_repeats': 50
              }

agent_params = {'agent_type': AgentType.CTDL_A2C,
                'bSOM': True,
                'SOM_alpha': .01,
                'SOM_sigma': .1,
                'SOM_sigma_const': .1,
                'Q_alpha': .9,
                'w_decay': 10,
                'TD_decay': 1,
                'SOM_size': 15,
                'e_trials': 200
                }