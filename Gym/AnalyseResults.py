from Gym.Functions.Parsers import ParseIntoDataframes
from Gym.Functions.Plotters import PlotComparisons

dir = 'ContinuousMountainCar'
to_compare = ['CTDL_A2C', 'A2C']

data_frames, labels = ParseIntoDataframes(dir, to_compare)

PlotComparisons(data_frames, labels)
