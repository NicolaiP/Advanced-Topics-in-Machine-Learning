"""
Nicolai F. Pedersen, nicped@dtu.dk
Script to plot the results of the pretrained CCA models
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code import settings
from python_code.utils.file_utils import make_folder, load_pickle_file
from python_code.utils.visualization_utils import plot_or_save

color_map = plt.get_cmap('Reds')
mapcolors = [color_map(int(x * color_map.N / 100)) for x in range(100)]


def load_normalized_face_landmarks():
    """
    Loads the locations of each of the 68 landmarks
    :return:
    """

    normalized_face_landmarks = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    return normalized_face_landmarks


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


def visual_landmarks_cca_heatmap(visual_cca, ax, title='CC'):
    """ Plots the visual loadings as ellipses on top of landmarks, to show their contribution to the CCA

    :param ax:
    :param visual_cca:
    :param title:
    :param subject_name:
    :param save_fig:
    :param save_path:
    :return:
    """

    # Normalize the loadings to sum to one.
    if len(visual_cca.shape) == 2:
        visual_cca = visual_cca.squeeze()
    visual_cca = abs(visual_cca)
    if len(visual_cca) == 204:
        visual_cca = np.array([visual_cca[::3], visual_cca[1::3], visual_cca[2::3]]).T
    visual_cca = visual_cca / np.sum(visual_cca)

    # Load the normalized landmarks
    landmarks = load_normalized_face_landmarks()
    landmarks -= np.mean(landmarks, axis=0)

    max_landmarks = np.max(landmarks, axis=0)
    min_landmarks = np.min(landmarks, axis=0)

    max_landmarks += 0.1
    min_landmarks -= 0.1
    landmarks[:, 1] = -landmarks[:, 1]

    # Define ellipses based on the importance in the loadings
    ells = [Ellipse(xy=landmarks[i, :], width=0.04, height=0.04, angle=0) for i in range(len(landmarks))]
    ells_center = [Ellipse(xy=landmarks[i, :], width=0.005, height=0.005, angle=0) for i in range(len(landmarks))]
    if len(visual_cca.shape) == 2:
        mean_visual_cca = np.mean(visual_cca, axis=1)
        color_sort = np.round((mean_visual_cca/max(mean_visual_cca))*len(mapcolors))-1
        # color_sort = np.argsort(np.argsort(np.mean(visual_cca, axis=1)))
    else:
        # color_sort = np.argsort(np.argsort(visual_cca))
        color_sort = np.round((visual_cca/max(visual_cca))*len(mapcolors))-1

    # Plots the ellipses
    for e, color_idx in zip(ells, color_sort):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)  # how transparent or pastel the color should be
        e.set_facecolor(mapcolors[int(color_idx)])

    # Plots the center of ellipses
    for e in ells_center:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)  # how transparent or pastel the color should be
        e.set_facecolor([1, 0, 0])

    ax.set_xlim(min_landmarks[0], max_landmarks[0])
    ax.set_ylim(-max_landmarks[1], -min_landmarks[0])
    ax.set(xticks=[], yticks=[], title=title)
    # plt.colorbar(color_map, ax1=ax1)


def visual_landmarks_cca_subplot(cca_weights, save_name='subject', save_path='plots/landmark_loadings', save_fig=False,
                                 comp_titles=[]):
    """ Subplots the visual loadings as ellipses on top of landmarks, to show their contribution to the CCA

    :param visual_cca:
    :param title:
    :param save_name:
    :param save_fig:
    :param save_path:
    :return:
    """

    # Number of components
    comp = cca_weights.shape[1]

    # Define the size of the subplot
    if 8 >= comp > 4:
        rows = 2
    elif 12 >= comp > 8:
        rows = 3
    elif comp > 12:
        rows = 4
    else:
        rows = 1
    columns = int(np.ceil(comp / rows))
    fig, axs = plt.subplots(rows, columns, figsize=(int(6 * columns), int(6 * rows)))

    if not comp_titles:
        comp_titles = list(range(1, comp + 1))

    for count, (visual_cca, ax) in enumerate(zip(cca_weights.T, axs.flat)):
        visual_landmarks_cca_heatmap(visual_cca, ax, title='cc ' + str(comp_titles[count]))

    plot_or_save(save_fig=save_fig, save_path=save_path, save_name=save_name)

    return


class DataLoader(object):

    def __init__(self, dataset='actors', visual_feature='landmarks_3d', audio_feature='mod_filter', motion=False,
                 video_norm=True, frame_norm=False):

        self.dataset = dataset
        self.visual_feature = visual_feature
        self.audio_feature = audio_feature
        self.motion = motion
        self.video_norm = video_norm
        self.frame_norm = frame_norm

        valid_dataset = ('actors', 'grid', 'lrs3')

        if dataset not in valid_dataset:
            raise ValueError('Unknown dataset, should be one of: {}'.format(valid_dataset))

        if dataset == 'grid':
            self.data_path = settings.grid_path_init()
            self.fps = 25
        elif dataset == 'actors':
            self.data_path = settings.actors_path_init()
            self.fps = 30
        elif dataset == 'lrs3':
            self.data_path = settings.lrs3_path_init()
            self.fps = 25

        self.data_dict = self.load_hdf_dataset()

    def load_hdf_dataset(self):
        '''
        Loads in either actors dataset or grid dataset in a dictionary.
        :param dataset:
        :param visual_feature:
        :param audio_feature:
        :return:
        '''

        if self.dataset == 'grid':
            hdf_file = h5py.File(os.path.join(self.data_path, 'grid.hdf5'), 'r')
            movies_grid = list(set([key[:-6] for key in hdf_file.keys()]))

            # Create dictionary for each subject with a visual feature and audio feature
            data_dict = {subject: {self.visual_feature: list(), self.audio_feature: list()} for subject in movies_grid}
            movies_pr_subject = 1
            for movie in hdf_file.keys():
                if len(data_dict[movie[:-6]][self.visual_feature]) < movies_pr_subject:
                    try:
                        data_dict[movie[:-6]][self.visual_feature].append(
                            self.visual_norm(hdf_file[movie][self.visual_feature][()]))
                        data_dict[movie[:-6]][self.audio_feature].append(hdf_file[movie][self.audio_feature][()])
                    except:
                        NotImplemented
                else:
                    continue
        elif self.dataset == 'actors':
            hdf_file = h5py.File(os.path.join(self.data_path, 'actors.hdf5'), 'r')

            # Create dictionary for each subject with a visual feature and audio feature
            data_dict = {subject: {self.visual_feature: list(), self.audio_feature: list()} for subject in
                         hdf_file.keys()}
            for movie in hdf_file.keys():
                try:
                    data_dict[movie][self.visual_feature].append(
                        self.visual_norm(hdf_file[movie][self.visual_feature][()]))
                    data_dict[movie][self.audio_feature].append(hdf_file[movie][self.audio_feature][()])
                except:
                    NotImplemented
        elif self.dataset == 'lrs3':
            hdf_file = h5py.File(os.path.join(self.data_path, 'lrs3.hdf5'), 'r')

            # Create dictionary for each subject with a visual feature and audio feature
            data_dict = {subject: {self.visual_feature: list(), self.audio_feature: list()}
                         for subject in hdf_file.keys()
                         if self.visual_feature in hdf_file[subject] and self.audio_feature in hdf_file[subject]}

            for movie in data_dict.keys():
                try:
                    data_dict[movie][self.visual_feature].append(
                        self.visual_norm(hdf_file[movie][self.visual_feature][()]))
                    data_dict[movie][self.audio_feature].append(hdf_file[movie][self.audio_feature][()])
                except:
                    NotImplemented
        else:
            raise ValueError('Unknown dataset!')

        for movie in data_dict.keys():
            data_dict[movie][self.visual_feature] = np.array(np.vstack(data_dict[movie][self.visual_feature]),
                                                             dtype=np.float32)
            data_dict[movie][self.audio_feature] = np.array(np.vstack(data_dict[movie][self.audio_feature]),
                                                            dtype=np.float32)

        return data_dict

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
        Normalizes visual 3d landmarks across the frames and/or calculates motion vector
        :param data: visual 3d landmarks
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
        ##TODO use https://en.wikipedia.org/wiki/Procrustes_analysis

        data = data.reshape((data_shape[0], -1))
        if self.motion:
            # Get motion vector based on distance between adjacent frames
            # First the face landmarks are reshaped such that they are: number of frames x 68 x 3
            data = data.reshape(data_shape)
            face_landmarks_dist = np.zeros(data.shape[:-1])
            face_landmarks_dist[1:] = np.linalg.norm(data[1:, :, :] - data[:-1, :, :], axis=2)
            data = face_landmarks_dist
        return np.array(data, dtype=np.float32)


def plot_pretrained_cca(audio_feature='mod_filter', visual_feature='landmarks_3d', save_fig=True, dataset='grid',
                        video_norm=True, frame_norm=False):
    """
    Makes plot of the summary face and the frequency content in reconstructed audio from the significant components
    :param audio_feature:
    :param visual_feature:
    :param save_fig:
    :param dataset:
    :param video_norm:
    :param frame_norm:
    :return:
    """

    dl = DataLoader(dataset=dataset, visual_feature=visual_feature, audio_feature=audio_feature,
                    video_norm=video_norm, frame_norm=frame_norm)

    if video_norm:
        audio_feature = audio_feature + '_video_norm'
    if frame_norm:
        audio_feature = audio_feature + '_frame_norm'

    save_kwargs = {'save_fig': save_fig, 'save_path': dl.data_path + '/plots/result_plots/'}
    make_folder(save_kwargs['save_path'])

    load_path = dl.data_path + '/plots/comp_permutation/'

    # Load the CCA model
    cca = load_pickle_file(load_path + '_'.join(('best_cca', visual_feature, audio_feature)))

    frames_per_movie = int(dl.fps * 3)
    a_train, v_train = dl.get_all(divisible_by=frames_per_movie)

    # Compute cross loadings
    a_train -= cca.x_mean_
    a_train /= cca.x_std_
    v_train -= cca.y_mean_
    v_train /= cca.y_std_

    # Load the data for plotting
    perm_dict = load_pickle_file(load_path + '_'.join(('perm_dict', visual_feature, audio_feature)))
    significan_comp = [ii for ii in perm_dict['significant_comp']]
    comp_titles = [ii + 1 for ii in significan_comp]

    frames, feature_len = np.shape(a_train)
    comp = len(significan_comp)

    n_movies = int(frames / frames_per_movie)
    frames = int(n_movies * frames_per_movie)
    a_train = a_train[:frames]
    v_train = v_train[:frames]
    trans_a, trans_v = cca.transform(a_train, v_train)
    a_weights = cca.x_weights_.T

    if 8 >= comp > 4:
        rows = 2
    elif 12 >= comp > 8:
        rows = 3
    elif comp > 12:
        rows = 4
    else:
        rows = 1

    columns = int(np.ceil(comp / rows))
    plt.figure(figsize=(int(6 * columns), int(6 * rows)))

    # Compute mean FFT of the reconstructed audio for each significant component
    all_mean_freq = []
    for count, (comp, comp_t) in enumerate(zip(significan_comp, comp_titles), start=1):
        recon_audio = np.dot(np.expand_dims(trans_a[:, comp], -1), np.expand_dims(a_weights[comp, :], 0))
        recon_audio1 = np.reshape(recon_audio, (n_movies, frames_per_movie, -1))
        freq = dl.fps / frames_per_movie * np.array(range(int(frames_per_movie)))

        # Compute the frequency for each one landmark but for all the videos
        freq_landmarks = np.zeros(recon_audio1.shape[:2])
        for jj in range(recon_audio1.shape[0]):
            freq_landmarks[jj, :] = np.fft.fftshift(np.fft.fft(recon_audio1[jj, :, 0]))

        mean_freq_landmarks = abs(np.mean(freq_landmarks, 0))
        all_mean_freq.append(mean_freq_landmarks)

    all_mean_freq1 = np.array(all_mean_freq).T
    scaled_mean_freq_landmarks1 = all_mean_freq1 / np.max(all_mean_freq1, 0)
    scaled_mean_freq_landmarks1 = np.mean(scaled_mean_freq_landmarks1, 1)
    scaled_mean_freq_landmarks1 = scaled_mean_freq_landmarks1 / np.max(scaled_mean_freq_landmarks1, 0)
    plt.plot(freq[int(frames_per_movie / 2):] - min(freq[int(frames_per_movie / 2):]),
             scaled_mean_freq_landmarks1[int(frames_per_movie / 2):])
    plt.xticks(np.arange(0, int(dl.fps / 2), step=1))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('normalized amplitude')
    plt.xlim([0, int(dl.fps / 2)])
    plt.title('Audio summary frequency')
    plt.grid()
    plot_or_save(save_name='audio_summary_frequencies_' + audio_feature, **save_kwargs)

    # Plot summary face loadings from significant components
    visual_loadings = abs(cca.y_loadings_[:, significan_comp])
    mean_visual_loadings = np.expand_dims(np.mean(visual_loadings, 1), -1)
    fig, ax = plt.subplots()
    visual_landmarks_cca_heatmap(mean_visual_loadings, ax, title=' '.join(('Summary face', audio_feature)))
    plot_or_save(save_name='_'.join(('summary_face', visual_feature, audio_feature)), **save_kwargs)

    print('Done')

    return


if __name__ == '__main__':
    plot_pretrained_cca()
    print('done')
