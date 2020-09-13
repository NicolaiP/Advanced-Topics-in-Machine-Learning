"""
Nicolai F. Pedersen, nicped@dtu.dk
This script runs the optimization of the regularized CCA model using Bayesian Optimization.
The script can be called from cmd line and inputs can be given depending on the dataset or features.
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code import settings
from python_code.models.cca import CCA
from python_code.utils.file_utils import make_folder, save_as_pickle, load_pickle_file
from python_code.utils.visualization_utils import plot_or_save


parser = argparse.ArgumentParser(description="Define training parameters for the model.")
parser.add_argument("-d", "--dataset", default='actors', choices=['grid', 'actors'])
parser.add_argument("-v", "--video_norm", default='True', choices=['True', 'False'])
parser.add_argument("-f", "--frame_norm", default='False', choices=['True', 'False'])
parser.add_argument("-a", "--audio_feature", default='mod_filter', choices=['mod_filter', 'mfcc_feat'])

args = parser.parse_args()


def augment_landmarks(landmarks, n_augment=10):
    """
    Given landmarks this function rotates the landmarks around y and z axis n_augment times.

    :param landmarks: Flatten 3d landmarks with shape n_frames x 204 (68 x 3)
    :param n_augment: number of times the landmarks should be augmented
    :return:
    """
    rot_z = lambda X, a: np.dot(X, np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]],
                                            dtype=np.float32))
    rot_y = lambda X, a: np.dot(X, np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]],
                                            dtype=np.float32))

    landmarks = np.reshape(landmarks, (len(landmarks), 68, 3))
    new_landmarks = []
    new_landmarks.extend(landmarks)
    angle_range = np.pi / 8
    np.random.seed(0)
    angles = np.random.uniform(-angle_range, angle_range, (n_augment, 2))
    for z_angle, y_angle in angles:
        new_landmarks.extend(rot_y(rot_z(landmarks, z_angle), y_angle))

    return np.array(new_landmarks, dtype=np.float32).reshape(-1, 68 * 3)


class DataLoader(object):
    """
    The DataLoader class loads and normalizes the data.
    """

    def __init__(self, dataset='actors', visual_feature='landmarks_3d', audio_feature='mod_filter', motion=False,
                 video_norm=True, frame_norm=False):

        self.dataset = dataset
        self.visual_feature = visual_feature
        self.audio_feature = audio_feature
        self.motion = motion
        self.video_norm = video_norm
        self.frame_norm = frame_norm

        valid_dataset = ('actors', 'grid')

        if dataset not in valid_dataset:
            raise ValueError('Unknown dataset, should be one of: {}'.format(valid_dataset))

        if dataset == 'grid':
            self.data_path = settings.grid_path_init()
            self.fps = 25
        elif dataset == 'actors':
            self.data_path = settings.actors_path_init()
            self.fps = 30

        self.data_dict = {}

    def load_hdf_dataset(self):
        '''
        Loads in either actors dataset or grid dataset in a dictionary.
        :param dataset:
        :param visual_feature:
        :param audio_feature:
        :return:
        '''

        if self.dataset in ['grid', 'actors']:
            hdf_file = h5py.File(os.path.join(self.data_path, self.dataset + '.hdf5'), 'r')

            # Create dictionary for each subject with a visual feature and audio feature
            self.data_dict = {subject: {self.visual_feature: list(), self.audio_feature: list()} for subject in
                         hdf_file.keys()}
            for movie in hdf_file.keys():
                try:
                    self.data_dict[movie][self.visual_feature] = self.visual_norm(hdf_file[movie][self.visual_feature][()])
                    self.data_dict[movie][self.audio_feature] = hdf_file[movie][self.audio_feature][()]
                except:
                    NotImplemented
        else:
            raise ValueError('Unknown dataset!')

    def get_all(self, divisible_by=1):
        """
        Gets all the features. divisible_by, makes sure that the data can be partitioned into divisible by frames.
        :param divisible_by:
        :return:
        """
        v_train = []
        a_train = []
        for train_key in self.data_dict.keys():
            frames_in_video = self.data_dict[train_key][self.audio_feature].shape[0]
            frames_2_include = frames_in_video - (frames_in_video % divisible_by)
            v_train.extend(self.data_dict[train_key][self.visual_feature][:frames_2_include, :])
            a_train.extend(self.data_dict[train_key][self.audio_feature][:frames_2_include, :])
        a_train = np.array(a_train)
        v_train = np.array(v_train)
        return a_train, v_train

    def visual_norm(self, data):
        '''
        Normalizes visual 3d landmarks across the videos, frames and/or calculates motion vector
        :param data: visual 3d landmarks
        :param video_norm: normalize landmarks across videos
        :param video_norm: normalize landmarks across frames
        :param motion: If True the motion vector is calculated based on distance between adjacent frames
        :return:
        '''
        data_shape = data.shape
        if self.video_norm and self.frame_norm:
            raise ValueError('Impossible to use video norm and frame norm at the same time')

        if self.video_norm:
            data = data.reshape((-1, 3))
            mean_dir = np.mean(data, axis=0)
            std_dir = np.std(data, axis=0)
            data -= mean_dir
            data = data / std_dir
        elif self.frame_norm:
            data1 = np.zeros_like(data)
            for count, frame in enumerate(data):
                mean_dir = np.mean(frame, axis=0)
                std_dir = np.std(frame, axis=0)
                frame -= mean_dir
                data1[count, :, :] = frame / std_dir
            data = data1

        data = data.reshape((data_shape[0], -1))
        if self.motion:
            # Get motion vector based on distance between adjacent frames
            # First the face landmarks are reshaped such that they are: number of frames x 68 x 3
            data = data.reshape(data_shape)
            face_landmarks_dist = np.zeros(data.shape[:-1])
            face_landmarks_dist[1:] = np.linalg.norm(data[1:, :, :] - data[:-1, :, :], axis=2)
            data = face_landmarks_dist
        return np.array(data, dtype=np.float32)


def comp_permutation_reg(audio_feature='mod_filter', visual_feature='landmarks_3d', save_fig=True, dataset='grid',
                         motion=False, video_norm=True, frame_norm=False):
    """
    This function runs the optimization of thre regularized CCA using Bayesian Optimization
    :param audio_feature:
    :param visual_feature:
    :param save_fig:
    :param dataset:
    :param motion:
    :param video_norm:
    :param frame_norm:
    :return:
    """

    # id_vec vector to extract correct indecies.
    dl = DataLoader(dataset=dataset, visual_feature=visual_feature, audio_feature=audio_feature, motion=motion,
                    video_norm=video_norm, frame_norm=frame_norm)

    dl.load_hdf_dataset()

    save_kwargs = {'save_fig': save_fig, 'save_path': dl.data_path + '/plots/comp_permutation/'}
    if save_fig:
        make_folder(save_kwargs['save_path'])

    if video_norm:
        audio_feature = audio_feature + '_video_norm'
    if frame_norm:
        audio_feature = audio_feature + '_frame_norm'
    if motion:
        audio_feature = audio_feature + '_motion'

    path_joiner = lambda x_str: save_kwargs['save_path'] + '_'.join((x_str, visual_feature, audio_feature))

    num_perm = 1000

    all_cv = {'reg_audio': [], 'reg_visual': [], 'largest_area': [], 'significant_comp': []}

    # Get training data
    frames_per_movie = dl.fps

    a_train, v_train = dl.get_all(divisible_by=frames_per_movie)
    frames, feature_len = np.shape(a_train)
    n_components = feature_len

    # Reshape data such that each movie consist of one second. This allows for permutation
    n_movies = int(frames / frames_per_movie)
    frames = int(n_movies * frames_per_movie)
    a_train = a_train[:frames]
    v_train = v_train[:frames]
    a_train1 = np.reshape(a_train, (n_movies, frames_per_movie, -1))

    x_space = [10 ** -5, 10 ** 0]
    y_space = [10 ** -5, 10 ** 0]
    space = [Real(x_space[0], x_space[1], "log-uniform", name='reg1'),
             Real(y_space[0], y_space[1], "log-uniform", name='reg2')]

    perm_dict = {'significant_comp': [], 'min_func': [], 'perm_corr': [], 'true_corr': []}
    gp_dict = {'x_iters': [], 'func_vals': []}
    objective_count = []

    @use_named_args(space)
    def objective(**params):

        print('Run ' + str(len(objective_count) + 1) + ' out of ' + str(n_calls))

        # Extract regularization
        audio_reg = params['reg1']
        visual_reg = params['reg2']

        if len(objective_count) < len(gp_dict['x_iters']):
            # Checks whether results have been saved from previous run, if so it uses these values.
            loss_function = gp_dict['func_vals'][len(objective_count)]
        elif [audio_reg, visual_reg] in gp_dict['x_iters']:
            # Checks whether the parameters have been evaluated before, if so it uses these values.
            loss_function = [gp_dict['func_vals'][count] for count, ii in enumerate(gp_dict['x_iters']) if ii == [audio_reg, visual_reg]][0]
        else:
            # Train CCA on non-permutated data
            cca = CCA(n_components=n_components, reg1=audio_reg, reg2=visual_reg)
            cca.fit(a_train, v_train)
            x_scores, y_scores = cca.transform(a_train, v_train)
            true_corr = np.diag(np.corrcoef(x_scores.T, y_scores.T)[n_components:, :n_components])

            # Run CCA on permutated data
            perm_corr = np.zeros((num_perm, n_components))
            for ii in range(num_perm):
                # THE PERMUTATION AND RESHAPING HAS BEEN DOUBLED CHECKED AND IT WORKS LIKE A CHARM
                cca_rand = CCA(n_components=n_components, reg1=audio_reg, reg2=visual_reg)
                rand = np.random.permutation(n_movies)
                a_train_rand = a_train1[rand, :, :]
                a_train_rand = np.reshape(a_train_rand, (frames, -1))
                cca_rand.fit(a_train_rand, v_train)
                x_scores, y_scores = cca_rand.transform(a_train_rand, v_train)
                perm_corr[ii, :] = np.diag(np.corrcoef(x_scores.T, y_scores.T)[n_components:, :n_components])

            perm_corr = np.array(perm_corr)

            # Identify the permutations below the significance level
            perm = 1 - sum(true_corr - perm_corr > 0) / num_perm
            perm_good = np.where(perm < 0.05)[0]

            # Get the median of the random permutations
            median_corr = np.median(perm_corr, axis=0)

            # As the function seeks to minimizes we convert to minus area
            loss_function = -sum(true_corr[perm_good] - median_corr[perm_good])

        gp_dict['x_iters'].append([audio_reg, visual_reg])
        gp_dict['func_vals'].append(loss_function)
        save_as_pickle(gp_dict, path_joiner('gp_dict'))
        df = pd.DataFrame(gp_dict)
        df.to_csv(path_joiner('gp_dict') + '.csv', index=False)

        if loss_function <= min(gp_dict['func_vals']):
            print('Save because it is best')
            save_as_pickle(cca, path_joiner('best_cca'))
            perm_dict['significant_comp'] = perm_good
            perm_dict['min_func'] = loss_function
            perm_dict['perm_corr'] = perm_corr
            perm_dict['true_corr'] = true_corr
            save_as_pickle(perm_dict, path_joiner('perm_dict'))

        objective_count.append(1)

        return loss_function

    n_calls = 10
    n_random_starts = 5

    try:
        gp_dict = load_pickle_file(path_joiner('gp_dict'))
        perm_dict = load_pickle_file(path_joiner('perm_dict'))
        x0 = gp_dict['x_iters']
        y0 = gp_dict['func_vals']
        function_gp = gp_minimize(objective, space, x0=x0, y0=y0, n_calls=n_calls, n_jobs=1, verbose=False,
                                  n_random_starts=n_random_starts, random_state=0)
    except FileNotFoundError:
        # Description of bayesian optimization https://distill.pub/2020/bayesian-optimization/
        function_gp = gp_minimize(objective, space, n_calls=n_calls, n_jobs=1, verbose=False,
                                  n_random_starts=n_random_starts, random_state=0)

    print('Best corr: ' + str(-function_gp['fun']) + '\n')
    print('with audio reg = ' + str(function_gp['x'][0]) + ', visual reg = ' + str(function_gp['x'][1]) + '\n\n')
    audio_reg = function_gp['x'][0]
    visual_reg = function_gp['x'][1]

    if save_fig:
        all_cv['reg_audio'].append(audio_reg)
        all_cv['reg_visual'].append(visual_reg)
        all_cv['largest_area'].append(-function_gp['fun'])
        all_cv['significant_comp'].append(perm_dict['significant_comp'])
        df = pd.DataFrame(all_cv)
        df.to_csv(path_joiner('area') + '.csv', index=False)
        save_as_pickle(perm_dict, path_joiner('perm_dict'))

    # Plot the permutations
    cca_perm_plot(audio_feature, visual_feature, audio_reg, visual_reg, save_kwargs, **perm_dict)

    return


def cca_perm_plot(audio_feature, visual_feature, audio_reg, visual_reg, save_kwargs,
                  perm_corr, true_corr, significant_comp, min_func):
    """
    Plots the correlation for each component along with the corresponding permutations. Further, the significant
    components are highlighted.
    :param audio_feature:
    :param visual_feature:
    :param audio_reg:
    :param visual_reg:
    :param save_kwargs:
    :param perm_corr:
    :param true_corr:
    :param significant_comp:
    :param min_func:
    :return:
    """

    n_components = len(true_corr)
    plt.figure(figsize=(int(12), int(5)))

    # Plot the first line, to only get one legend
    plt.plot(range(1, n_components + 1), perm_corr[0, :].T, linewidth=0.5, zorder=0, label='Permutated order')
    plt.plot(range(1, n_components + 1), perm_corr[1:, :].T, linewidth=0.5, zorder=0)
    plt.plot(range(1, n_components + 1), true_corr.T, c='orange', linewidth=3, zorder=1, label='Correct order')
    plt.scatter(significant_comp + 1, true_corr[significant_comp], c='r', linewidths=0.5, zorder=2, label='Significant components')
    plt.xlim([1, n_components])
    plt.ylim([0, 0.5])
    plt.xticks(range(1, n_components + 1, 1))
    plt.xlabel('Components')
    plt.ylabel('Correlation')
    plt.legend(framealpha=1, frameon=True)
    plt.grid()
    plt.title(f'Area = {abs(min_func):.4f} - RegA = {audio_reg:.0e}, RegV = {visual_reg:.0e}')
    plot_or_save(save_name='_'.join(('comp_perm', audio_feature, visual_feature)), **save_kwargs)


if __name__ == '__main__':
    audio_feature1 = args.audio_feature
    dataset = args.dataset
    if args.video_norm == 'True':
        video_norm = True
        frame_norm = False
    elif args.frame_norm == 'True':
        frame_norm = True
        video_norm = False
    else:
        video_norm = True
        frame_norm = False

    comp_permutation_reg(audio_feature=audio_feature1, dataset=dataset, video_norm=video_norm, frame_norm=frame_norm)
