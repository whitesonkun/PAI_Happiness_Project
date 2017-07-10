import mne


def mne_raw_apply_fun(eeg: mne.io.RawArray,fun,channels:list=None, **kwargs):
    out = []
    for idx, chan in enumerate(eeg.ch_names):
        if (channels is None) or (chan in channels):
            out.append(fun(eeg[idx, :][0], **kwargs))
    return out