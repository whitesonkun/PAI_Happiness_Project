## Imports
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from detect_peaks import detect_peaks
from scipy.stats import zscore
from scipy.ndimage.filters import uniform_filter
import warnings
from scipy.interpolate import interp1d
from scipy.io import loadmat
import datetime
from biosppy.signals import bvp
import glob
from parse import *


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X * X, window_size, mode='reflect')
    return np.sqrt(c2 - c1 * c1)


## Read EEG
def process_events(edf_filename: str, timing_filename: str, init_slice: int) -> object:
    sample_rate = 128

    channels_to_exclude = ['COUNTER', 'INTERPOLATED', 'RAW_CQ', 'GYROX', 'GYROY', 'MARKER', 'MARKER_HARDWARE', 'SYNC',
                           'TIME_STAMP_s', 'TIME_STAMP_ms', 'CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 'CQ_P7',
                           'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8', 'CQ_AF4', 'CQ_CMS', 'CQ_DRL']

    channel_map = {'O1': 'F7',
                   'P7': 'Sync1',
                   'T7': 'Fz',
                   'F7': 'F5',
                   'AF3': 'F8',
                   'FC5': 'C3',
                   'F3': 'Cz',
                   'O2': 'Sync2',
                   'P8': 'P7',
                   'T8': 'P3',
                   'F8': 'Pz',
                   'AF4': 'P4',
                   'FC6': 'P8',
                   'F4': 'Oz'}

    raw_edf = mne.io.read_raw_edf(edf_filename, stim_channel=None, exclude=channels_to_exclude)
    try:
        raw_edf.load_data()
    except IndexError:
        warnings.warn("Failed to load data, skipping subject")
        return
    raw_edf.rename_channels(channel_map)
    raw_edf = mne.set_bipolar_reference(raw_edf, "Sync1", "Sync2", "SyncRR")
    sync_channel0 = raw_edf.copy().pick_channels(["SyncRR"])
    data, times = sync_channel0[:]  # Extract data
    data = data.transpose()
    data = zscore(data)  # normalize
    times -= init_slice
    data = data[times > 0]
    times = times[times > 0]
    # plt.plot(np.asarray(times), data)
    # plt.show()
    diff_data_raw = np.insert(np.diff(data, axis=0), 0, 0)
    diff_data = diff_data_raw.copy()
    threshold_moving = window_stdev(diff_data, 5000) * 3.5  # Calculate a moving threshold by
    diff_data[np.where((diff_data > -threshold_moving) & (diff_data < threshold_moving))] = 0  # Threshold
    diff_data[np.where(diff_data < 0)] = -diff_data[np.where(diff_data < 0)]  # Rectify

    # plt.plot(np.asarray(times), diff_data)
    eeg_times = detect_peaks(diff_data, mph=0.2)
    eeg_times_edge = np.where(diff_data_raw[eeg_times] > 0, 1, 0)  # rising edge is 1
    rising = eeg_times[eeg_times_edge == 1]
    falling = eeg_times[eeg_times_edge == 0]
    # plt.plot(rising / sample_rate, np.ones(rising.shape), '*')
    # plt.plot(falling / sample_rate, np.ones(falling.shape), '*')
    eeg_times_secs = eeg_times / sample_rate

    event_data = pd.read_excel(timing_filename)
    event_data = event_data[['label', 'getSecsLogTime', 'realTime']]
    event_data.columns = ["label", "matlab_time", 'datetime']
    event_data['matlab_time'] -= event_data['matlab_time'][0]  # Start at zero
    event_data['eeg_time'] = np.nan

    recogRespTimes = event_data.label.apply(lambda lab: 'recogRespTime' not in lab)
    event_data_working = add_expected_transitions(event_data[recogRespTimes])
    max_expected_lag = 5  # Initial Differences between matlab and eeg should not be more than 5

    event_data_working['label_generic'] = event_data_working['label'].apply(lambda lab: lab[0:lab.rfind("_")])

    # from collections import defaultdict
    # time_testing_dict = defaultdict(list)

    event_data.set_index('label', inplace=True, drop=False)

    matlab_labels = event_data_working['label_generic'].unique()
    for label in matlab_labels:
        timing_data_slice = event_data_working[label == event_data_working['label_generic']]
        print("For", label, "attempting to find", len(timing_data_slice.matlab_time), "events")
        # plt.figure()
        # plt.plot(timing_data_slice.matlab_time, np.ones(len(timing_data_slice.matlab_time)), '*')
        # a = eeg_times_secs[(eeg_times_secs < (timing_data_slice.matlab_time.iloc[-1] + max_expected_lag)) & (
        # eeg_times_secs > (timing_data_slice.matlab_time.iloc[0] - max_expected_lag))]
        # plt.plot(a, np.ones(len(a)), '*')
        # plt.show()
        expected_diff = np.diff(timing_data_slice.matlab_time)
        if label in ['syncDataLight_BeforeInter_flipTime']:
            sync_idx, sync_lag = find_template_in_sequence(eeg_times_secs, eeg_times_edge, expected_diff,
                                                           timing_data_slice.trans_bool, 0,
                                                           eeg_times_secs[-1] / 4)
            if sync_idx is None:
                warnings.warn('Did not fine sync signal. Terminating....')
                return
            eeg_times_secs = eeg_times_secs[sync_idx[0]:] - sync_lag
            eeg_times_edge = eeg_times_edge[sync_idx[0]:]
            lag_eeg_to_matlab = sync_lag
            print('Found start at', sync_lag)
            sync_idx = range(0, len(sync_idx))
        else:
            sync_idx, sync_lag = find_template_in_sequence(eeg_times_secs, eeg_times_edge, expected_diff,
                                                           timing_data_slice.trans_bool,
                                                           timing_data_slice.matlab_time.iloc[0] - max_expected_lag,
                                                           timing_data_slice.matlab_time.iloc[-1] + max_expected_lag)
            # Linearly increasing lag
            max_expected_lag = 5 + 0.01 * timing_data_slice.matlab_time.iloc[-1]

        if sync_idx is None:
            print('Did not find all points for', label)
            continue

        # for key, val in time_testing.items():
        #     time_testing_dict[key].extend(val)

        # print("Found", len(sync_idx), "events at", sync_lag)
        for lab, val in zip(timing_data_slice.label, eeg_times_secs[sync_idx]):
            event_data.set_value(lab, 'eeg_time', val)

    event_data = add_event_data_missing_sync(event_data)

    event_data = interpolate_missing(event_data)

    # times -= lag_eeg_to_matlab
    # data = data[times > 0]
    # times = times[times > 0]
    # plot_sync_times(data, times, event_data)

    # TODO remove numbers from label to create a version for easy ERP events

    return event_data


def add_event_data_missing_sync(event_data):
    for idx, og_row in event_data.iterrows():
        if 'syncDataLight' in og_row.label:
            new_row = og_row.copy()
            new_row.label = new_row.label.replace('flipTime', 'sound')
            new_row.matlab_time = og_row.matlab_time + 0.5
            event_data.loc[event_data.shape[0]] = new_row

    return event_data


def add_expected_transitions(event_data_matlab):
    """
    Creates a simple copy of the event data from matlab  (label, matlab_time, and expected trans columns)
        and populates it with the expected transitions based on a transition template file

    Args:
        event_data_matlab (ndarray): The event data from matlab

    Returns:
        ndarray: the event data from matlab with expected transitions
    """

    sync_template = pd.read_excel("syncTemplate.xlsx")

    num_events_in_sync_template = 2 * len(sync_template[(sync_template.label != 'append')
                                                        & (sync_template.label != 'prepend')])
    event_data_with_expected_trans = pd.DataFrame(columns=('label', 'matlab_time', 'trans'))
    idx = 0

    for idx_sync_times, og_sync_row in event_data_matlab.iterrows():
        lab_to_match = og_sync_row.label.replace('BeforeInter_', '')
        lab_to_match = lab_to_match.replace('AfterInter_', '')
        try:
            loc_in_template = np.flatnonzero(lab_to_match == sync_template.label)[0]
        except IndexError:
            if "recogRespTime" not in lab_to_match:  # we know we will not find recogRespTime
                print('Did not find', lab_to_match)
        if sync_template.label[loc_in_template - 1] == 'prepend':
            event_data_with_expected_trans.loc[idx] = ['prepend', np.nan, sync_template.trans[loc_in_template - 1]]
            idx += 1

        if 'replace' == sync_template.notes[loc_in_template]:
            event_data_with_expected_trans.loc[idx] = ['replace', np.nan, sync_template.trans[loc_in_template]]
            idx += 1
        else:
            event_data_with_expected_trans.loc[idx] = [og_sync_row.label, og_sync_row.matlab_time,
                                                       sync_template.trans[loc_in_template]]
            idx += 1

        if sync_template.label[loc_in_template + 1] == 'append':
            event_data_with_expected_trans.loc[idx] = ['append', np.nan, sync_template.trans[loc_in_template + 1]]
            idx += 1

    # Deal with the fact the emotionMem find times are in random order, append to last one
    for wordFindMatch in ['wordFind_BeforeInter_findTimes', 'wordFind_AfterInter_findTimes']:
        find_times = event_data_with_expected_trans.label.apply(lambda lab: wordFindMatch in lab)
        last_find_time_labels = event_data_with_expected_trans.label[find_times]
        try:
            last_loc = np.flatnonzero(event_data_with_expected_trans.label == last_find_time_labels.iloc[-1])[0]
        except IndexError:
            warnings.warn('Could not find the last "find_time", this subject likely has missing events')
        line = pd.DataFrame({'label': 'append', 'matlab_time': np.nan, 'trans': 'White'}, index=[last_loc])
        event_data_with_expected_trans = pd.concat([event_data_with_expected_trans.iloc[:last_loc + 1],
                                                    line,
                                                    event_data_with_expected_trans.iloc[last_loc + 1:]]).reset_index(
            drop=True)

    colormap = {'Black': 1, 'White': 0}
    event_data_with_expected_trans['trans_bool'] = event_data_with_expected_trans.trans.map(colormap)
    trans_diffs = np.insert(np.diff(event_data_with_expected_trans.trans_bool), 0, 1)
    event_data_with_expected_trans = event_data_with_expected_trans.loc[trans_diffs.nonzero()[0], :]

    event_data_with_expected_trans = event_data_with_expected_trans[
        (event_data_with_expected_trans.label != 'append')
        & (event_data_with_expected_trans.label != 'prepend')
        & (event_data_with_expected_trans.label != 'replace')]

    if num_events_in_sync_template != len(event_data_with_expected_trans):  # FIXME - always reports a dif...
        warnings.warn(
            "This sub did not complete {} events".format(
                num_events_in_sync_template - len(event_data_with_expected_trans)))

    return event_data_with_expected_trans


def plot_sync_times(raw_data, raw_times, event_data):
    """Plots the sync labels along with the matching times found from eeg data"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(raw_times, raw_data)

    for idx, event in event_data.iterrows():
        ax.annotate(event.name, xy=(event.eeg_time, 0), xytext=(event.matlab_time, plt.ylim()[1]),
                    arrowprops=dict(facecolor='grey', width=1), rotation=45
                    )
    plt.show()


def interpolate_missing(event_data):
    """

    Missing events that did not have a correct sync signal associated with them are added
    by interpolating from nearby points

    """
    missing_idxs = np.isnan(event_data.eeg_time)
    # Interp fails when the value to be interpolated occurs at a position higher than the range of existing data...
    last_found_idx = np.flatnonzero(~missing_idxs)[-1]
    missing_idxs[last_found_idx:] = False
    if last_found_idx != len(missing_idxs) - 1:
        warnings.warn("Could not interpolate final points, was the final event found?")
    interp_f = interp1d(event_data.matlab_time[~missing_idxs], event_data.eeg_time[~missing_idxs])
    event_data.loc[event_data.index[missing_idxs], 'eeg_time'] = interp_f(event_data[missing_idxs].matlab_time)
    return event_data.sort_values('eeg_time')


def find_template_in_sequence(data_seq, data_edge, template, template_edge, start_looking=0,
                              end_looking=None):
    """ 
    Find a specific pattern of numbers in an sorted array. Small deviations from the pattern are tolerated.
    
    Args:
        data_seq (ndarray): The array to find the matching pattern in
        template (ndarray): Pattern of differences to find in data_seq array
        tolerance (float): how much each point in the template may be off by and a match still found
        start_looking (int): value to begin searching at
        end_looking (int): value to end searching at 

    Returns:
        int: the value where the template is first found
        int: the index where the template is first found
    """

    warnings.simplefilter('always', UserWarning)

    if end_looking is None:
        end_looking = max(data_seq)

    vals_ok_low = data_seq <= end_looking
    vals_ok_high = data_seq >= start_looking
    num_low_values_to_drop = sum(~vals_ok_high)
    data_seq = data_seq[vals_ok_low & vals_ok_high]
    data_edge = data_edge[vals_ok_low & vals_ok_high]
    for idx, point in enumerate(data_seq):
        if data_edge[idx] != template_edge.iloc[0]:
            continue  # Make sure first transition is in correct direction
        matched_idxs = [idx]
        next_point = point
        found_pattern = True
        time_testing = {'drift': [], 'time_diff': []}  # For testing time drift only...
        for interval in template:
            next_point += interval
            diff_to_target = np.abs(data_seq - next_point)
            diff_to_target[data_edge != template_edge.iloc[len(matched_idxs)]] = np.inf
            closest_point_idx = np.argmin(diff_to_target)
            closest_point = np.min(diff_to_target)
            if closest_point > 0.08 + 0.01 * interval:  # The 0.01 was found by regressing expected interval vs drift, 0.08 is added to give a little buffer
                # print('Error, closest point is:', closest_point, 'and only found', len(matched_idxs), 'matching points')
                found_pattern = False
                break
            next_point = data_seq[closest_point_idx]
            matched_idxs.append(closest_point_idx)
            time_testing['drift'].append(closest_point)
            time_testing['time_diff'].append(interval)

        if found_pattern:
            return np.array(matched_idxs) + num_low_values_to_drop, point  # , time_testing

    warnings.warn("Pattern not found")
    return (None, None)


def load_feel_data(sub_id):  # TODO machine learning to maximize this shit
    feel_loc = "../Data/FeelData/EEGStudy1/Feel_Data/"
    feel_data = loadmat(feel_loc + "Subject_" + str(sub_id) + ".mat")
    ts = feel_data['numStart'][0]
    bvp_data = [i[0] for i in feel_data['denB']]
    temp_data = [i[0] for i in feel_data['denT']]
    gsr_data = [i[0] for i in feel_data['denG']]
    secs = int(ts[5])
    ms = 1000 * (int(ts[5]) - secs)
    start_time = datetime.datetime(int(ts[0]), int(ts[1]), int(ts[2]), int(ts[3]), int(ts[4]), secs, ms)
    time_data = [datetime.time(int(t_raw[0]), int(t_raw[1]), int(t_raw[2])) for t_raw in feel_data['ACQ']]
    bvp_sig = bvp.bvp(signal=bvp_data, sampling_rate=20, show=True)
    # bvp_sig.find_onsets()

    print('Function Exit')


def load_eeg_data(sub_id: int) -> object:
    edf_loc = "../Data/EEG/"
    timestamp_loc = "../Data/FeelData/EEGStudy1/Timestamps/"

    init_dict = {2017: 100}  # skip the first part of the record
    subs_to_ignore = [i for i in range(2001, 2012)]
    subs_to_ignore.extend([2013, 2021, 2039, 2059, 2066])

    good_subs = [2014, 2017, 2018, 2019, 2022, 2024, 2028, 2029, 2030, 2031, 2037, 2041, 2044, 2046, 2048, 2058, 2060,
                 2065, 2068, 2071, 2073]

    subs_with_noise = [2060]

    bad_sync_channel = [2027, 2034, 2035, 2038, 2043, 2045, 2061, 2064, 2067, 2069, 2072]

    print('--------------------Parsing EEG for subject', sub_id, '-------------------')

    # EEG loading
    if sub_id in subs_to_ignore:
        print('Skipping, known bad file...')
        return
    # if subId[0] in good_subs:
    #     print('Skipping, known good file...')
    #     continue
    if sub_id in bad_sync_channel:
        print('Skipping, bad sync channel...')
        return
    if sub_id in init_dict:
        init = init_dict[sub_id]
    else:
        init = 0
    return process_events(edf_loc + "PAI_" + str(sub_id) + ".edf",
                          timestamp_loc + "PAI_RawData_Subject" + str(sub_id) + "_EEGSync_Timestamps.xlsx", init)


if __name__ == "__main__":

    behav_loc = "../Data/Behavioral/EEGStudy1/"

    # ideally import counterbalance of good and bad files
    sub_ids = [2019]
    sub_data = dotdict({})

    for sub_id in sub_ids:
        # sub_data.feel_data = load_feel_data(sub_id)
        sub_data.eeg_data = load_eeg_data(sub_id)

    print(sub_data) #TODO run analysis, Move to modules
