import numpy as np
import matplotlib.pyplot as plt

from GridWorld.Agents.CTDL.SOMLayer import SOMLayer


class SOM(object):

    def __init__(self, directory, maze_width, maze_height, input_dim, map_size, learning_rate, sigma, sigma_const):

        self.directory = directory
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.SOM_layer = SOMLayer(np.amax([maze_width, maze_height]), input_dim, map_size, learning_rate, sigma, sigma_const)

        self.location_counts = np.zeros((maze_height, maze_width))

        return

    def Update(self, state, best_unit, reward_value):

        self.SOM_layer.Update(state, best_unit, reward_value)

        return

    def GetOutput(self, state):

        best_unit = self.SOM_layer.GetBestUnit(state)

        return best_unit


    def PlotResults(self, plot_num):

        self.PlotMap(plot_num)
        self.PlotLocations(plot_num)

        return

    def PlotMap(self, plot_num):

        width = np.unique(self.SOM_layer.units['xy']).shape[0]
        height = width
        im_grid = np.zeros((width, height, 3))

        for i in range(width * height):
            image = np.zeros(3)
            image[:2] = self.SOM_layer.units['w'][i, :]
            image = np.clip(np.array(image) / np.amax([self.maze_width, self.maze_height]), 0, 1)
            im_grid[self.SOM_layer.units['xy'][i, 0], self.SOM_layer.units['xy'][i, 1], :] = image
        plt.figure()
        plt.imshow(im_grid)
        plt.savefig(self.directory + 'SOM%06d.pdf' % plot_num)
        plt.close()

        return

    def PlotLocations(self, plot_num):

        im_grid = np.zeros((self.maze_height, self.maze_width))

        for i in range(self.SOM_layer.num_units):
            y = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1)))
            x = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1)))
            im_grid[y, x] = 1

        plt.figure()
        plt.imshow(im_grid)
        plt.savefig(self.directory + 'SOMLocations%06d.pdf' % plot_num)
        plt.close()

        np.save(self.directory + 'SOMLocations', im_grid)

        return

    def RecordLocationCounts(self):

        for i in range(self.SOM_layer.num_units):
            y = int(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1))
            x = int(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1))
            self.location_counts[y, x] += 1

        return