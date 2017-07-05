import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from detect_peaks import detect_peaks
from scipy.stats import zscore
from scipy.ndimage.filters import uniform_filter
import warnings
import datetime
from scipy.interpolate import interp1d
from yetti_utils import dotdict

def plot_potentially_bad_channels(eeg):
    high_amp_channels = flag_channels_with_high_amplitude(eeg)
    eeg.plot()
    plt.show()


def flag_channels_with_high_amplitude(eeg):
    channel_list = []
    return channel_list