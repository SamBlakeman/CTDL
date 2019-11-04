from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.Plotters import PlotRevaluationComparisons, PlotMeanSOMLocations

dir = 'RevaluationSweep'
to_compare = ['CTDL', 'DQN']

data_frames, labels = ParseIntoDataframes(dir, to_compare)

PlotRevaluationComparisons(data_frames, labels)
PlotMeanSOMLocations('Results/' + dir + '/', data_frames[0])


