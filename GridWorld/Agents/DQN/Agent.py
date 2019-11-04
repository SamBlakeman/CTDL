import matplotlib.pyplot as plt
import numpy as np
import pickle

from GridWorld.Agents.DQN.Memory import Memory
from GridWorld.Agents.DQN.QTargetGraph import QTargetGraph
from GridWorld.Agents.DQN.QGraph import QGraph


class Agent(object):

    def __init__(self, directory, maze_params, agent_params):

        self.directory = directory
        self.maze_width = maze_params['width']
        self.maze_height = maze_params['height']

        self.minibatch_size = 32

        self.q_graph = QGraph(4, self.directory, self.maze_width)
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory, self.maze_width)

        self.memory = Memory()

        self.discount_factor = 0.99
        self.epsilon = 0
        self.final_epsilon = .9
        self.num_epsilon_trials = agent_params['e_trials']
        self.epsilon_increment = self.final_epsilon / self.num_epsilon_trials

        self.c = 10000
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

    def NewMaze(self, directory):

        self.directory = directory
        self.q_graph.directory = directory
        self.UpdateTargetGraph()

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_Qvalue = None
        self.bStart_learning = False

        return


    def UpdateQGraph(self):

        self.ci += 1
        if (self.ci >= self.c):
            self.UpdateTargetGraph()

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

    def UpdateTargetGraph(self):
        print('Loading New target Graph')
        self.ci = 0
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory, self.maze_width)
        return

    def SelectAction(self, state):

        if(np.random.rand() > self.epsilon):
            action = np.random.randint(4)
        else:
            action = np.argmax(np.squeeze(np.array(self.q_graph.GetActionValues(state))))

        self.prev_action = action
        self.prev_state = np.copy(state)

        return action


    def PlotResults(self):

        plt.figure()
        plt.plot(self.results['rewards'])
        found_goal = np.where(np.array(self.results['rewards']) > 0)
        if (found_goal):
            for loc in found_goal[0]:
                plt.axvline(x=loc, color='g')
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def PlotValueFunction(self):

        up_value_function = np.zeros((self.maze_height, self.maze_width))
        down_value_function = np.zeros((self.maze_height, self.maze_width))
        left_value_function = np.zeros((self.maze_height, self.maze_width))
        right_value_function = np.zeros((self.maze_height, self.maze_width))

        for row in range(self.maze_height):
            for col in range(self.maze_width):
                action_values = np.squeeze(np.array(self.q_graph.GetActionValues(np.array([[row, col]]))))
                up_value_function[row, col] = action_values[0]
                down_value_function[row, col] = action_values[1]
                left_value_function[row, col] = action_values[2]
                right_value_function[row, col] = action_values[3]

        fig, axes = plt.subplots(2, 2)

        im = axes[0, 0].imshow(up_value_function, cmap='hot')
        axes[0, 0].set_title('Up Value Function')

        im = axes[0, 1].imshow(down_value_function, cmap='hot')
        axes[0, 1].set_title('Down Value Function')

        im = axes[1, 0].imshow(left_value_function, cmap='hot')
        axes[1, 0].set_title('Left Value Function')

        im = axes[1, 1].imshow(right_value_function, cmap='hot')
        axes[1, 1].set_title('Right Value Function')

        for axis in axes.ravel():
            axis.set_xticklabels([])
            axis.set_xticks([])
            axis.set_yticklabels([])
            axis.set_yticks([])

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(self.directory + 'ValueFunction%06d.pdf' % self.plot_num)
        plt.close()

        self.plot_num += 1

        return


