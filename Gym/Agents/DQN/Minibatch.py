
class MiniBatch(object):

    def __init__(self):
        self.prev_states = []
        self.actions = []
        self.rewards = []
        self.states = []
        self.bTrial_over = []