import numpy as np

class SOMLayer():

    def __init__(self, input_dim, size, learning_rate, sigma, sigma_const):

        self.size = size
        self.num_units = size * size
        self.num_weights = input_dim

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sigma_const = sigma_const

        self.units = {'xy': [], 'w': []}
        self.ConstructMap()

        return

    def ConstructMap(self):

        x = 0
        y = 0

        # Construct map
        for u in range(self.num_units):

            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.randn(self.num_weights))#np.random.randn(self.num_weights))

            x += 1

            if (x >= self.size):
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])

        return

    def Update(self, state, best_unit, reward_value):

        diffs = self.units['xy'] - self.units['xy'][best_unit, :]
        location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))
        neighbourhood_values = np.exp(-np.square(location_distances) / (2.0 * (self.sigma_const + (reward_value * self.sigma))))

        self.units['w'] += (reward_value * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])

        return

    def GetBestUnit(self, state):

        best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)

        return best_unit