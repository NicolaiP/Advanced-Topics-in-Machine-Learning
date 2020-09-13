"""
Nicolai F. Pedersen, nicped@dtu.dk
Some useful functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
sys.path.append("../")  # go to parent dir


def save_as_pickle(variable_name, save_name):
    """Saves variable as pickle file.
    # Arguments
        save_name: Name of file.
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        save_name = dataPath + 'predictionsResNet50ADAM_lr0001_decay0005'
        file_utils.save_as_pickle(preds, save_name)
    """
    f = open(save_name + '.pckl', 'wb')
    pickle.dump(variable_name, f)
    f.close()


def load_pickle_file(path):
    """Loads pickle file.
    # Arguments
        path: Path to file.
    # Returns
        var: Loaded variables.
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        fileName = dataPath + 'predictionsResNet50ADAM_lr0001_decay0005'
        var = file_utils.load_pickle_file(path)
    """
    if path.split('.')[-1] == 'pckl':
        var = pickle.load(open(path, 'rb'))
    else:
        var = pickle.load(open(path + '.pckl', 'rb'))
    return var


def make_folder(data_path):
    '''
    Function that creates a folder if it doesn't exist
    :param data_path:
    :return:
    '''
    if not os.path.exists(data_path):
        os.makedirs(data_path)


if __name__ == '__main__':
    make_folder()
