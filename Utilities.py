import os
import numpy as np
import shutil

def RecordSettings(directory, maze_params, agent_params):
    file = open(directory + 'Settings.txt', 'w')

    for key, value in maze_params.items():
        file.write(key + ': ' + str(value) + '\n')

    for key, value in agent_params.items():
        file.write(key + ': ' + str(value) + '\n')

    file.close()
    return

def RecordSetting(directory, key, value):
    file = open(directory + 'Settings.txt', 'a')
    file.write(key + ': ' + str(value) + '\n')
    file.close()
    return
