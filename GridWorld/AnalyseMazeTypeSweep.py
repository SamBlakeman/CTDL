from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.Plotters import PlotComparisons, PlotMeanSOMLocations

dir = 'MazeTypeSweep'
to_compare = ['CTDL', 'DQN']

data_frames, labels = ParseIntoDataframes(dir, to_compare)

PlotComparisons('type', data_frames, labels)
PlotMeanSOMLocations('Results/' + dir + '/', data_frames[0])


