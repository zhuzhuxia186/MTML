import logging
import os
import os.path
import time
import argparse
import sys
import pandas as pd
import torch
from braindecode.datautil.signal_target import SignalAndTarget
from scipy import signal
import numpy as np
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
import scipy.io as sio
from os.path import join
import logging
import os.path
from collections import OrderedDict
import sys
import numpy as np
import torch
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import train_test_split
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def mergeSet(set1, set2):
    tempX = np.concatenate((set1.X, set2.X))
    tempY = np.concatenate((set1.y, set2.y))
    return SignalAndTarget(tempX, tempY)

def mergeChannel(sat1,sat2):

    A = torch.from_numpy(sat1.X)
    B = torch.from_numpy(sat2.X)

    B=B.unsqueeze(dim=3)
    tempX=torch.cat((A,B),dim=3)
    tempX = tempX.detach().numpy()

    return SignalAndTarget(tempX,sat1.y)

def preprocessing(subject_id,save_path,low_hz,high_hz):


    low_cut_hz = low_hz
    high_cut_hz = high_hz

    factor_new = 1e-3
    init_block_size = 1000

    train_ival = [-500, 4000]  #
    test_ival = [-500, 4000]  #

    # Data loading
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(save_path, train_filename)
    test_filepath = os.path.join(save_path, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # Preprocessing
    # train_cnt
    # train_cnt = train_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    # assert len(train_cnt.ch_names) == 3
    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    # bandpass
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    # test_cnt
    # test_cnt = test_cnt.pick_channels(['EEG-C3', 'EEG-C4', 'EEG-Cz'])
    # assert len(test_cnt.ch_names) == 3
    test_cnt = test_cnt.drop_channels(['EOG-central', 'EOG-right', 'STI 014', 'EOG-left'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, train_ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, test_ival)

    train_set.X = np.expand_dims(train_set.X, axis=1)
    train_set.y = train_set.y.astype(np.int64)
    test_set.X = np.expand_dims(test_set.X, axis=1)
    test_set.y = test_set.y.astype(np.int64)

    return train_set,test_set


def read_shu_2a(hz_tuple,read_file_path,sub_id):
    #sub_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    l_hz,h_hz = hz_tuple
    train_set_all,test_set_all = preprocessing(sub_id,read_file_path,l_hz,h_hz)

    return train_set_all, test_set_all







