import mne
from joblib import Parallel, delayed
import numpy as np


def mne_raw_apply_fun(eeg: mne.io.RawArray, fun, channels:list=None, chan_types_to_include: list=['eeg'], n_jobs=1, **kwargs):
    to_run = []
    for idx, chan in enumerate(eeg.ch_names):
        if (channels is None) or (chan in channels):
                if mne.io.pick.channel_type(eeg.info, idx) in chan_types_to_include:
                    if chan not in eeg.info['bads']:
                        to_run.append(eeg[idx, :][0])

    if n_jobs > 1:
        return Parallel(n_jobs=n_jobs)(
            delayed(fun)(chan_data, **kwargs) for chan_data in to_run)


    else:
        out = []
        for chan_data in to_run:
                out.append(fun(chan_data, **kwargs))
        return out


def filter_by_frequency_bands(eeg: mne.io.RawArray, filt_bands=[(1, 3), (3, 10), (10, 20), (20, 60)], hilbert=False):
    """
    Applies bandpass filtering in the specified frequency bands to all eeg channels, appends to eeg data.
    Operates in place.
    """
    for fband in filt_bands:
        raw_filt = eeg.copy()
        fband_name = '_' + str(fband[0]) + "Hz-" + str(fband[1]) + 'Hz'
        channel_name_remap = {chan: chan+fband_name for chan in eeg.ch_names}
        raw_filt.filter(*fband, h_trans_bandwidth='auto', l_trans_bandwidth='auto',
                        filter_length='auto', phase='zero')  # Filter all EEG Data
        raw_filt.rename_channels(channel_name_remap)

        if hilbert:
            raw_hilb = raw_filt.copy()
            raw_hilb.apply_hilbert()
            raw_hilb.apply_function(np.abs)
            channel_name_remap = {chan: chan + '_env' for chan in raw_filt.ch_names}
            raw_hilb.rename_channels(channel_name_remap)
            return raw_filt, raw_hilb #TODO APPEND THIS SHIT

        return raw_filt
