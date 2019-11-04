import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn.preprocessing

from Gym.Agents.A2C.ACGraph import ACGraph
from Gym.Agents.CTDL_A2C.SOM import DeepSOM


class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.bSOM = agent_params['bSOM']
        self.directory = directory
        self.action_maxs = env_params['action_maxs']
        self.action_mins = env_params['action_mins']
        self.input_dim = env_params['state_dim']

        self.ac_graph = ACGraph(self.input_dim, self.action_mins, self.action_maxs, self.directory)
        self.ac_graph.SaveGraphAndVariables()

        if (self.bSOM):
            self.CreateSOM(agent_params)

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        self.discount_factor = 0.99
        self.epsilon = 1

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_Vvalue = None
        self.bStart_learning = False

        state_space_samples = np.array(
            [env_params['env_obj'].observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_space_samples)

        return

    def CreateSOM(self, agent_params):

        self.SOM = DeepSOM(self.directory, self.input_dim, agent_params['SOM_size'],
                           agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                           agent_params['SOM_sigma_const'])
        self.V_alpha = agent_params['Q_alpha']
        self.VValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size']))

        return

    def ScaleState(self, state):
        scaled = self.scaler.transform([state])
        return scaled

    def Update(self, reward, state, bTrial_over):

        state = self.ScaleState(np.squeeze(state))
        self.RecordResults(bTrial_over, reward)

        if (self.bStart_learning):
            self.UpdateACGraph(bTrial_over, reward, state)

        action = self.SelectAction(state)

        if (not self.bStart_learning):
            self.bStart_learning = True

        return action


    def RecordResults(self, bTrial_over, reward):

        self.trial_reward += reward
        self.trial_length += 1
        if (bTrial_over):
            self.results['rewards'].append(self.trial_reward)
            print('Cumulative Episode Reward: ' + str(self.trial_reward))
            self.trial_reward = 0

            self.results['lengths'].append(self.trial_length)
            self.trial_length = 0

        return

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

    def GetVValues(self, state, critic_value):

        best_unit = self.SOM.GetOutput(state)
        som_value = self.VValues[best_unit]
        w = self.GetWeighting(best_unit, state)
        state_value = (w * som_value) + ((1 - w) * critic_value)

        return state_value

    def UpdateACGraph(self, bTrial_over, reward, state):

        prev_state_value = self.ac_graph.GetStateValue(self.prev_state)
        target = self.GetTargetValue(bTrial_over, reward, state)

        delta = target - prev_state_value
        self.ac_graph.GradientDescentStep(self.prev_state, self.prev_action, target, delta)

        if (self.bSOM):
            self.UpdateSOM(target)

        return

    def UpdateSOM(self, target):

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        delta = np.exp(np.abs(target -
                              np.squeeze(self.ac_graph.GetStateValue(
                                  self.prev_state))) / self.TD_decay) - 1

        delta = np.clip(delta, 0, 1)
        self.SOM.Update(self.prev_state, prev_best_unit, delta)

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        w = self.GetWeighting(prev_best_unit, self.prev_state)
        self.VValues[prev_best_unit] += self.V_alpha * w * (
                target - self.VValues[prev_best_unit])

        return

    def GetTargetValue(self, bTrial_over, reward, state):

        critic_value = np.squeeze(np.array(self.ac_graph.GetStateValue(state)))

        if(self.bSOM):
            state_value = self.GetVValues(state, critic_value)
        else:
            state_value = critic_value

        if (bTrial_over):
            target = reward
        else:
            target = reward + (state_value * self.discount_factor)

        return target


    def SelectAction(self, state):

        action = self.ac_graph.GetAction(state)
        self.prev_action = action
        self.prev_state = np.copy(state)

        return action


    def PlotResults(self):
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(self.results['rewards'])
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

