import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from yetti_utils import dotdict

def extract_erps(eeg: mne.io.RawArray, events: pd.DataFrame, plot: bool=True):
    """Extract and plot ERP from EEG data"""
    mne_events, event_map = convert_to_mne_events(events)
    epochs_params = dict(events=mne_events, event_id=event_map, tmin=-0.2, tmax=0.5, reject=dict(eeg=180e-6))
    evoked = mne.Epochs(eeg, **epochs_params).average()

    evoked.plot()
    evoked.plot_topomap(times=[0.1, 0.2, 0.3], size=3.)

    plt.show(block=False)


def pick_event_type(events, type):
    print('')

def convert_to_mne_events(events):
    event_map = {event: int(idx+1) for (idx, event) in enumerate(np.unique(events.index.values))}

    mne_events = pd.DataFrame(columns=['event_sample', 'previous_event_id', 'event_id'], dtype=int)
    mne_events['event_sample'] = events.astype(int)
    mne_events['event_id'] = events.index.map(lambda lab: event_map[lab])
    mne_events['previous_event_id'] = int(0)
    return mne_events.values, event_map
