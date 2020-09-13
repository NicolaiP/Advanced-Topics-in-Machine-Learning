from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt

from python_code.utils.file_utils import make_folder


def plot_or_save(save_fig=False, save_path='/', save_name='average'):

    if save_fig:
        make_folder(save_path)
        fname = save_name + '.png'
        print('Saving frame', fname)
        plt.savefig(os.path.normpath(os.path.join(save_path, fname)), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == '__main__':
    plot_or_save()
