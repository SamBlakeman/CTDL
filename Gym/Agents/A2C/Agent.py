import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn.preprocessing

from Gym.Agents.A2C.ACGraph import ACGraph


class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.directory = directory
        self.action_maxs = env_params['action_maxs']
        self.action_mins = env_params['action_mins']
        self.input_dim = env_params['state_dim']

        self.ac_graph = ACGraph(self.input_dim, self.action_mins, self.action_maxs, self.directory)
        self.ac_graph.SaveGraphAndVariables()

        self.discount_factor = 0.99
        self.epsilon = 1

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.bStart_learning = False

        state_space_samples = np.array(
            [env_params['env_obj'].observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_space_samples)

        return

    def ScaleState(self, state):
        scaled = self.scaler.transform([state])
        return scaled

    def Update(self, reward, state, bTrial_over):

        state = self.ScaleState(np.squeeze(state))
        self.RecordResults(bTrial_over, reward)

        if (self.bStart_learning):
            self.UpdateACGraph(reward, state, bTrial_over)

        action = self.SelectAction(state)

        if (not self.bStart_learning):
            self.bStart_learning = True

        return action

    def UpdateACGraph(self, reward, state, bTrial_over):

        state_value = self.ac_graph.GetStateValue(state)
        prev_state_value = self.ac_graph.GetStateValue(self.prev_state)

        if(bTrial_over):
            target = reward
        else:
            target = reward + self.discount_factor * np.squeeze(state_value)

        delta = target - prev_state_value
        self.ac_graph.GradientDescentStep(self.prev_state, self.prev_action, target, delta)

        return

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


    def SelectAction(self, state):

        action = self.ac_graph.GetAction(state)
        self.prev_action = action
        self.prev_state = np.copy(state)

        return action


    def PlotResults(self):

        plt.figure()
        plt.plot(self.results['rewards'])
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

