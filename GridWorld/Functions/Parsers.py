import os
import pickle
import numpy as np
import pandas as pd


def ParseIntoDataframes(dir, to_compare):
    folders = os.listdir('Results/' + dir)
    data_frames = []
    labels = []
    sorted_folders = [[] for i in range(to_compare.__len__())]
    for folder in folders:

        if (folder == '.DS_Store' or folder == '.keep'):
            pass

        else:
            files = os.listdir('Results/' + dir + '/' + folder)

            if ('.DS_Store' in files):
                files.remove('.DS_Store')

            file = open('Results/' + dir + '/' + folder + '/Settings.txt', 'r')
            settings = file.readlines()
            file.close()

            for setting in settings:
                vals = setting.strip('\n').split(': ')

                if (vals[0] == 'agent_type'):
                    try:
                        ind = np.where(np.array(to_compare) == vals[1].split('.')[1])[0][0]
                        sorted_folders[ind].append(folder)
                    except:
                        pass
    for model, folders in zip(to_compare, sorted_folders):
        data_frames.append(ParseDataFrame(folders, dir))
        labels.append(model)

    return data_frames, labels

def ParseDataFrame(folders, dir):

    results_dict = {'dir': [], 'rewards': [], 'lengths': [], 'maze': []}

    for folder in folders:

        try:
            with open('Results/' + dir + '/' + folder + '/Results.pkl', 'rb') as handle:
                dict = pickle.load(handle)

            results_dict['dir'].append(folder)

            results_dict['rewards'].append(dict['rewards'])
            results_dict['lengths'].append(dict['lengths'])

            file = open('Results/' + dir + '/' + folder + '/Settings.txt', 'r')
            settings = file.readlines()
            file.close()

            for setting in settings:
                vals = setting.split(': ')

                if (vals[0] not in results_dict):
                    results_dict[vals[0]] = []

                try:
                    results_dict[vals[0]].append(float(vals[1]))
                except:
                    results_dict[vals[0]].append(vals[1])

            results_dict['maze'].append(np.load('Results/' + dir + '/' + folder + '/Maze.npy'))
        except:
            pass

    df = pd.DataFrame.from_dict(results_dict)

    return df