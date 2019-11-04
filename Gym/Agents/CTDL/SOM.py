
from Gym.Agents.CTDL.SOMLayer import SOMLayer


class DeepSOM(object):

    def __init__(self, directory, input_dim, map_size, learning_rate, sigma, sigma_const):

        self.directory = directory
        self.SOM_layer = SOMLayer(input_dim, map_size, learning_rate, sigma, sigma_const)

        return

    def Update(self, state, best_unit, reward_value):

        self.SOM_layer.Update(state, best_unit, reward_value)

        return

    def GetOutput(self, state):

        best_unit = self.SOM_layer.GetBestUnit(state)

        return best_unit