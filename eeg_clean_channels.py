import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from detect_peaks import detect_peaks
from scipy.stats import zscore
from scipy.ndimage.filters import uniform_filter
import warnings
import datetime
from scipy.interpolate import interp1d
from yetti_utils import *
from mne_utils import mne_raw_apply_fun

def plot_potentially_bad_channels(eeg, impedence=None):
    channel_info = pd.DataFrame(columns=['channel_names', 'sd'])

    std_per_channel = mne_raw_apply_fun(eeg, channels=eeg.ch_names[0:-1], fun=np.std, axis=1)
    channel_info['sd'] = sp.stats.zscore([std[0] for std in std_per_channel])
    channel_info['channel_names'] = eeg.ch_names[0:-1]
    channel_info = channel_info.set_index('channel_names', drop=True)

    if impedence:
        mean_impedance = np.array(mne_raw_apply_fun(impedence, fun=np.mean))
        imps = sp.stats.zscore(mean_impedance)
        for idx, chan in enumerate(impedence.ch_names):
            if chan[3:] in channel_info.index:
                channel_info.set_value(chan[3:], 'mean_impedance', imps[idx])


    df_to_heatmap(channel_info)

    eeg.plot()
    plt.show()
    print('Complete')

def df_to_heatmap(data, cmap=plt.cm.Blues):
    # Plot it out
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data.values, cmap=cmap, alpha=0.8)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(data.columns.values.tolist(), minor=False)
    ax.set_yticklabels(data.index, minor=False)

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

