import matplotlib.pyplot as plt
import numpy as np
import pickle

from Gym.Agents.CTDL.QGraph import QGraph
from Gym.Agents.CTDL.SOM import DeepSOM
from Gym.Agents.CTDL.QTargetGraph import QTargetGraph

class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.bSOM = agent_params['bSOM']
        self.directory = directory
        self.input_dim = env_params['state_dim']
        self.input_min = env_params['state_mins']
        self.input_max = env_params['state_maxs']
        self.num_actions = env_params['num_actions']

        self.q_graph = QGraph(self.input_dim, self.num_actions, self.directory)
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory)

        if(self.bSOM):
            self.CreateSOM(agent_params)

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        self.discount_factor = 0.99
        self.epsilon = 0
        self.final_epsilon = .9
        self.num_epsilon_trials = agent_params['e_trials']
        self.epsilon_increment = self.final_epsilon / self.num_epsilon_trials
        self.batch_size = 32

        self.c = 500
        self.ci = 0

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_Qvalue = None
        self.bStart_learning = False

        self.state_max = np.zeros(self.input_dim)
        self.state_min = np.zeros(self.input_dim)

        return

    def CreateSOM(self, agent_params):

        self.SOM = DeepSOM(self.directory, self.input_dim, agent_params['SOM_size'],
                           agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                           agent_params['SOM_sigma_const'])
        self.Q_alpha = agent_params['Q_alpha']
        self.QValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size'], self.num_actions))

        return

    def Update(self, reward, state, bTrial_over):

        for i, s in enumerate(state):
            if(s > self.state_max[i]):
                self.state_max[i] = s

            elif (s < self.state_min[i]):
                self.state_min[i] = s

        state = (state - self.state_min) / (self.state_max - self.state_min)


        if (bTrial_over and self.epsilon < self.final_epsilon):
            self.epsilon += self.epsilon_increment

        self.RecordResults(bTrial_over, reward)

        if(self.bStart_learning):
            self.UpdateQGraph(reward, state, bTrial_over)

        action = self.SelectAction(state)

        if(not self.bStart_learning):
            self.bStart_learning = True

        return action

    def RecordResults(self, bTrial_over, reward):

        self.trial_reward += reward
        self.trial_length += 1
        if (bTrial_over):
            self.results['rewards'].append(self.trial_reward)
            self.trial_reward = 0

            self.results['lengths'].append(self.trial_length)
            self.trial_length = 0

        return

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

    def GetQValues(self, state, q_graph_values):

        best_unit = self.SOM.GetOutput(state)
        som_action_values = self.QValues[best_unit, :]
        w = self.GetWeighting(best_unit, state)
        q_values = (w * som_action_values) + ((1 - w) * q_graph_values)

        return q_values


    def UpdateQGraph(self, reward, state, bTrial_over):


        self.ci += 1
        if (self.ci >= self.c):
            print('Loading New target Graph')
            self.ci = 0
            self.q_graph.SaveGraphAndVariables()
            self.q_target_graph = QTargetGraph(self.directory)


        target = self.GetTargetValue(bTrial_over, reward, state)

        self.q_graph.GradientDescentStep(np.expand_dims(self.prev_state, axis=0),
                                         np.expand_dims(self.prev_action, axis=0),
                                         np.expand_dims(target, axis=0))

        if(self.bSOM):
            self.UpdateSOM(target)

        return

    def UpdateSOM(self, target):

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        delta = np.exp(np.abs(target -
                              np.squeeze(self.q_graph.GetActionValues(
                                  np.expand_dims(self.prev_state, axis=0)))[self.prev_action]) / self.TD_decay) - 1

        delta = np.clip(delta, 0, 1)
        self.SOM.Update(self.prev_state, prev_best_unit, delta)

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        w = self.GetWeighting(prev_best_unit, self.prev_state)
        self.QValues[prev_best_unit, self.prev_action] += self.Q_alpha * w * (target - self.QValues[prev_best_unit, self.prev_action])
        self.Replay()

        return

    def GetTargetValue(self, bTrial_over, reward, state):

        q_graph_values = np.squeeze(np.array(self.q_target_graph.GetActionValues(np.expand_dims(state, axis=0))))

        if(self.bSOM):
            q_values = self.GetQValues(state, q_graph_values)

        else:
            q_values = q_graph_values

        max_q_value = np.amax(q_values)

        if (bTrial_over):
            target = reward

        else:
            target = reward + (max_q_value * self.discount_factor)

        return target

    def Replay(self):

        units = np.random.randint(0, self.SOM.SOM_layer.num_units, self.batch_size)
        actions = np.random.randint(0, self.num_actions, self.batch_size)

        self.q_graph.GradientDescentStep(self.SOM.SOM_layer.units['w'][units, :], actions, self.QValues[units, actions])

        return


    def SelectAction(self, state):

        q_graph_values = np.squeeze(np.array(self.q_graph.GetActionValues(np.expand_dims(state, axis=0))))

        if(self.bSOM):
            q_values = self.GetQValues(state, q_graph_values)
        else:
            q_values = q_graph_values

        if(np.random.rand() > self.epsilon):
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(q_values)

        self.prev_Qvalue = q_values[action]
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



