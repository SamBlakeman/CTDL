import numpy as np

from GridWorld.Agents.DQN.Minibatch import MiniBatch


class Memory(object):

    def __init__(self):

        self.capacity = 100000
        self.prev_states = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.bTrial_over = []

        return

    def RecordExperience(self, prev_state, state, action, reward, bTrial_over):

        self.prev_states.append(prev_state)
        self.states.append(state)
        self.rewards.append(reward)
        self.bTrial_over.append(bTrial_over)
        self.actions.append(action)

        if(self.rewards.__len__() > self.capacity):
            del self.prev_states[0]
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.bTrial_over[0]

        return


    def GetMinibatch(self, minibatch_size):

        minibatch = MiniBatch()
        experience_indices = np.random.randint(0, self.rewards.__len__(), minibatch_size)

        prev_states = []
        actions = []
        rewards = []
        states = []
        bTrial_over = []

        for i in experience_indices:

            prev_states.append(self.prev_states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            states.append(self.states[i])
            bTrial_over.append(self.bTrial_over[i])

        minibatch.prev_states = np.squeeze(np.array(prev_states, dtype=int))
        minibatch.actions = np.array(actions, dtype=int)
        minibatch.rewards = np.array(rewards, dtype=float)
        minibatch.states = np.squeeze(np.array(states, dtype=int))
        minibatch.bTrial_over = bTrial_over

        return minibatch
