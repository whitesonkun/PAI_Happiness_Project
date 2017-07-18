import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import zscore
import warnings
from yetti_utils import *
from mne_utils import *
import copy
import easygui
import time
import sys
from operator import itemgetter
from itertools import groupby

default_preprocessing_params = {'time':{
                                    'peak_to_peak_filter': {
                                        'peak_to_peak_filter.threshold': '2.7sd',
                                        'peak_to_peak_filter.length': 0.3,
                                        'peak_to_peak_filter.size_multiplier': 1.5,
                                        'peak_to_peak_filter.onset_shift': -0.15
                                    },
                                    'weighted_av_filter': {
                                        'weighted_av_filter.threshold': 0.00003,
                                        'weighted_av_filter.length': 0.2,
                                        'weighted_av_filter.size_multiplier': 1.1,
                                        'weighted_av_filter.onset_shift': -0.15,
                                    },
                                    'identity_filter': {
                                        'identity_filter.threshold'
                                        'identity_filter.length': 0.3,
                                        'identity_filter.size_multiplier': 1.5,
                                        'identity_filter.onset_shift': -0.15
                                    }
                                },
                                'freq':{
                                    'peak_to_peak_filter': {
                                        'peak_to_peak_filter.threshold': '2.7sd',
                                        'peak_to_peak_filter.length': 0.3,
                                        'peak_to_peak_filter.size_multiplier': 1.5,
                                        'peak_to_peak_filter.onset_shift': -0.15
                                    },
                                    'weighted_av_filter': {
                                        'weighted_av_filter.threshold': 0.00003,
                                        'weighted_av_filter.length': 0.2,
                                        'weighted_av_filter.size_multiplier': 1.1,
                                        'weighted_av_filter.onset_shift': -0.15,
                                    },
                                    'identity_filter': {
                                        'identity_filter.threshold'
                                        'identity_filter.length': 0.3,
                                        'identity_filter.size_multiplier': 1.5,
                                        'identity_filter.onset_shift': -0.15
                                    },
                                }
                            }


def apply_basic_preprocessing(eeg):
    eeg.set_eeg_reference()
    eeg.apply_proj()  # Apply re-ref
    eeg.filter(0.1, 60)


def reformat_data_for_quick_debug(eeg):
    eeg = eeg.crop(0, 300)
    return eeg
    #eeg.pick_channels(ch_names=['F8', 'F5'])


def interactive_channel_cleaner(eeg, impedance=None, filter_params=None):
    eeg = reformat_data_for_quick_debug(eeg)  # FIXME, Remove after debug
    apply_basic_preprocessing(eeg)
    sub_complete = False

    artifact_detector = ArtifactDetector(eeg)
    artifact_detector.plot_extra_channel_data(impedance)  # TODO split into its own step
    eeg.plot(title='Please select bad channels')
    plt.show()
    plt.close("all")

    # Calculate freq band
    eeg_chans = [chan for chan in eeg.ch_names if chan != 'events']
    eeg_freq, eeg_hilb = filter_by_frequency_bands(eeg, hilbert=True)
    freq_chans = [chan for chan in eeg_freq.ch_names if 'Hz' in chan]
    hilb_chans = [chan for chan in eeg_hilb.ch_names if 'hilb' in chan]
    eeg.add_channels([eeg_freq, eeg_hilb], force_update_info=True)

    # Apply interactive segment artifacting via filters
    peak_to_peak_params = {'fname': 'peak_to_peak_filter', #This would be better handled as it own class...
                           'params': {
                               'threshold': '2.7sd',
                               'length': 0.3,
                               'size_multiplier': 1.5,
                               'onset_shift': -0.15},
                           'channels': eeg_chans
                           }
    artifact_detector.add_threshold_filter('peak_to_peak_time', peak_to_peak_params)

    peak_to_peak_params = {'fname': 'peak_to_peak_filter',  # This would be better handled as it own class...
                           'params': {
                               'threshold': '2.5sd',
                               'length': 0.3,
                               'size_multiplier': 1.5,
                               'onset_shift': -0.15},
                           'channels': freq_chans
                           }
    artifact_detector.add_threshold_filter('peak_to_peak_freq', peak_to_peak_params)
    artifact_detector.run_filters()

    while not sub_complete:
        eeg.plot()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        threshold_filters_params, trigger_update = open_tune_params_box(artifact_detector.threshold_filters)
        plt.close("all")
        if trigger_update:
            updated = artifact_detector.update_filter_params(threshold_filters_params)
        if not updated:
            command = done_button()
            if command == 'Exit':
                return None
            if command == 'Save and Exit':
                sub_complete = True
                return filter_params


def done_button():
    return easygui.choicebox('Interactive Plotter', 'What do you want to do?',
                             ['Save and Exit', 'Exit', 'Refresh with new values'])


def open_tune_params_box(filter_params):
    """Opens an interactive box to change pre-processing params"""
    out = copy.deepcopy(filter_params)
    msg = "Add filter_params and push OK to refresh with new filter_params, cancel to ignore and keep old"
    title = "Tune filter_params:"
    for filter_name in filter_params:
        field_names = [filter_name + '.' + key for key in filter_params[filter_name]['params']]
        field_values = [value for key, value in
                        filter_params[filter_name]['params'].items()]  # we start with blanks for the values

    field_values = easygui.multenterbox(msg, title, field_names, field_values)
    if field_values is not None:
        for key, value in zip(field_names, field_values):
            filter_name = key.split('.')[0]
            param_name = key.split('.')[1]
            try:
                out[filter_name]['params'][param_name] = float(value)
            except ValueError:
                out[filter_name]['params'][param_name] = value
        return out, True
    else:
        return out, False


class ArtifactDetector:
    def __init__(self, eeg: mne.io.RawArray, threshold_filters: dict = None):

        self.eeg = eeg
        if self.eeg.annotations is None:
            self.eeg.annotations = mne.Annotations(onset=[], duration=[], description=[])
        if threshold_filters is None:
            self.threshold_filters = {}
        else:
            self.threshold_filters = threshold_filters

        self._sample_rate = eeg.info['sfreq']

    def add_threshold_filter(self, filter_name, filter_params):
        """Adds a threshold type artifacting filter to the artifact detector"""
        self.threshold_filters[filter_name] = filter_params

    def run_filters(self):
        for filter_ in self.threshold_filters:
            self._run_threshold_filter(filter_)

    def update_filter_params(self, new_params):
        funcs_to_rerun = []
        for filter_name in new_params:
            for param_name in new_params[filter_name]['params']:
                if self.threshold_filters[filter_name]['params'][param_name] != new_params[filter_name]['params'][param_name]:
                    self.threshold_filters[filter_name]['params'][param_name] = new_params[filter_name]['params'][param_name]
                    funcs_to_rerun.append(filter_name)

        funcs_to_rerun = np.unique(funcs_to_rerun)
        for func in funcs_to_rerun:
            self._remove_artifacts_associated_with_filter(func)
            self._run_threshold_filter(func)
        return len(funcs_to_rerun) > 0

    def _remove_artifacts_associated_with_filter(self, filter_to_remove):
        """Removes the artifacts found by the associated technique
        Args:
            :param filter_to_remove: the name of the function that generated annotations that are to be removed"""
        to_remove = []
        for idx, desc in enumerate(self.eeg.annotations.description):
            if filter_to_remove in desc:
                to_remove.append(idx)
        self.eeg.annotations.delete(to_remove)


        # Reref - Done

    # Threshold filter time domain: apply_func - apply custom functions to data
    # Threshold filter freq domain: https://martinos.org/mne/stable/auto_tutorials/plot_modifying_data_inplace.html
    # Sync channel can be removed via ICA or maybe SSP?
    # eyeblinks can be removed via ICA, and corr map, see http://martinos.org/mne/stable/auto_tutorials/plot_artifacts_correction_ica.html
    # Consider SSP over ICA when Signal 2 Noise is low: https://www.hindawi.com/journals/cin/2016/7489108/

    # Combining data: Normalize by calculating zscore average across all channels for each subject
    #



    def _run_threshold_filter(self, f_name):
        """Applies thresholding to the filter eeg channels to detect artifacts"""

        print(f_name)


        filtered_data = mne_raw_apply_fun(self.eeg,
                                          globals()[self.threshold_filters[f_name]['fname']],
                                          n_jobs=3,
                                          channels=self.threshold_filters[f_name]['channels'],
                                          sample_rate=self._sample_rate,
                                          filter_params=self.threshold_filters[f_name]['params'])

        for ch_idx, chan_data in enumerate(filtered_data):
            if 'sd' in self.threshold_filters[f_name]['params']['threshold']:
                mult = float(self.threshold_filters[f_name]['params']['threshold'].split('s')[0])
                artifacts = chan_data > mult*np.std(chan_data)
            else:
                artifacts = chan_data > self.threshold_filters[f_name]['params']['threshold']
            artifact_starts, artifact_ends, artifact_durations = runs_of_ones_array(artifacts)
            num_values = len(artifact_starts)
            if num_values > 0:
                onset = artifact_starts.transpose() / self._sample_rate
                duration = artifact_durations.transpose() * \
                    self.threshold_filters[f_name]['params']['size_multiplier'] / self._sample_rate
                label = np.tile(f_name + '.' + self.eeg.ch_names[ch_idx], [1, num_values])[0]
                self.eeg.annotations.append(onset, duration, label)

    def plot_extra_channel_data(self, impedance=None):
        channel_info = pd.DataFrame(columns=['channel_names', 'sd'])

        std_per_channel = mne_raw_apply_fun(self.eeg, channels=self.eeg.ch_names[0:-1], fun=np.std, axis=1)
        channel_info['sd'] = sp.stats.zscore([std[0] for std in std_per_channel])
        channel_info['channel_names'] = self.eeg.ch_names[0:-1]
        channel_info = channel_info.set_index('channel_names', drop=True)

        if impedance:
            mean_impedance = np.array(mne_raw_apply_fun(impedance, fun=np.mean))
            imps = sp.stats.zscore(mean_impedance)
            for idx, chan in enumerate(impedance.ch_names):
                if chan[3:] in channel_info.index:
                    channel_info.set_value(chan[3:], 'mean_impedance', imps[idx])

        df_to_heatmap(channel_info)


def weighted_av_filter(eeg_channel_data, sample_rate, filter_params):
    """
    Calculates a weighted moving average at every point along channel, params to control this are in
    pre-processing_params
    This function must be at top level (not class nested), otherwise it will not run correctly
    """
    print('applying weighted_av_filter')
    eeg_channel_data = eeg_channel_data[0]  # some strangeness with slicing ndarrays
    len_in_samples = int(filter_params['length'] * sample_rate)
    weights = np.array(range(len_in_samples))
    output = np.zeros(eeg_channel_data.shape[0])
    for i in range(len_in_samples, eeg_channel_data.shape[0]):
        items_in_bin = eeg_channel_data[(i - len_in_samples):i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output[i + int(filter_params['onset_shift'] * sample_rate)] = np.average(
                items_in_bin, weights=weights)
    return output


def peak_to_peak_filter(eeg_channel_data, sample_rate, filter_params):
    """
    Calculates the peak to peak difference within a window at every point along channel, params to control this are in
    filter_params
    This function must be at top level (not class nested), otherwise it will not run correctly
    """
    print('applying peak_to_peak_filter')
    eeg_channel_data = eeg_channel_data[0]  # some strangeness with slicing ndarrays
    len_in_samples = int(filter_params['length'] * sample_rate)
    output = np.zeros(eeg_channel_data.shape[0])
    for i in range(len_in_samples, eeg_channel_data.shape[0]):
        items_in_bin = eeg_channel_data[(i - len_in_samples):i]
        output[i + int(filter_params['onset_shift'] * sample_rate)] = np.abs(max(items_in_bin) - min(
            items_in_bin))
    return output


def identity_filter(eeg_channel_data, sample_rate, filter_params):
    """
    A 'filter' that does nothing
    This function must be at top level (not class nested), otherwise it will not run correctly
    """
    print('applying identity_filter')
    return eeg_channel_data[0]  # some strangeness with slicing ndarrays

    # def weighted_av_filter(eeg_channel_data, eeg, preprocessing_params):
    #     len_in_samples = int(preprocessing_params['weighted_av_thresholding']['length'] * eeg.info['sfreq'])
    #     weights = np.array(range(len_in_samples))
    #     output = np.zeros(len(eeg_channel_data))
    #     for i in range(len_in_samples, len(eeg_channel_data)):
    #         items_in_bin = eeg_channel_data[(i - len_in_samples):i]
    #         output[i] = np.average(items_in_bin, weights=weights)
    #     return output
