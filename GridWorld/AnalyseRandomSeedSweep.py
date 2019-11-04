from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.Plotters import PlotComparisons, PlotPairwiseComparison

dir = 'RandomSeedSweep'
to_compare = ['CTDL', 'DQN']

data_frames, labels = ParseIntoDataframes(dir, to_compare)

PlotComparisons('random_seed', data_frames, labels)
PlotPairwiseComparison(data_frames[0], data_frames[1], labels)
