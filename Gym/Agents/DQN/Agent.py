import matplotlib.pyplot as plt
import numpy as np
import pickle

from Gym.Agents.DQN.Memory import Memory
from Gym.Agents.DQN.QTargetGraph import QTargetGraph
from Gym.Agents.DQN.QGraph import QGraph


class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.directory = directory
        self.num_actions = env_params['num_actions']
        self.input_dim = env_params['state_dim']

        self.minibatch_size = 32

        self.q_graph = QGraph(self.input_dim, self.num_actions, self.directory)
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory)

        self.memory = Memory()

        self.discount_factor = 0.99
        self.epsilon = 0
        self.final_epsilon = .9
        self.num_epsilon_trials = agent_params['e_trials']
        self.epsilon_increment = self.final_epsilon / self.num_epsilon_trials

        self.c = 500
        self.ci = 0

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.bStart_learning = False

        return

    def Update(self, reward, state, bTrial_over):

        state = np.expand_dims(state, axis=0)

        if (bTrial_over and self.epsilon < self.final_epsilon):
            self.epsilon += self.epsilon_increment

        self.RecordResults(bTrial_over, reward)

        if(self.bStart_learning):
            self.memory.RecordExperience(self.prev_state, state, self.prev_action, reward, bTrial_over)
            self.UpdateQGraph()

        action = self.SelectAction(state)

        if (not self.bStart_learning):
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


    def UpdateQGraph(self):

        self.ci += 1
        if(self.ci >= self.c):
            print('Loading New target Graph')
            self.ci = 0
            self.q_graph.SaveGraphAndVariables()
            self.q_target_graph = QTargetGraph(self.directory)

        minibatch = self.memory.GetMinibatch(self.minibatch_size)
        max_action_values = np.amax(np.squeeze(np.array(self.q_target_graph.GetActionValues(minibatch.states))), axis=1)
        targets = np.zeros(minibatch.rewards.__len__())

        for i in range(targets.shape[0]):
            if(minibatch.bTrial_over[i]):
                targets[i] = minibatch.rewards[i]
            else:
                targets[i] = minibatch.rewards[i] + (max_action_values[i] * self.discount_factor)

        self.q_graph.GradientDescentStep(minibatch.prev_states, minibatch.actions, targets)

        return


    def SelectAction(self, state):

        if(np.random.rand() > self.epsilon):
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(np.squeeze(np.array(self.q_graph.GetActionValues(state))))

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

