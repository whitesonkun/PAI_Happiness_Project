## Imports
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from detect_peaks import detect_peaks
from scipy.stats import zscore
from scipy.ndimage.filters import uniform_filter
import warnings


def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X * X, window_size, mode='reflect')
    return np.sqrt(c2 - c1 * c1)


## Read EEG
def process_events(edf_filename, timing_filename):
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
    raw_edf.load_data()
    raw_edf.rename_channels(channel_map)
    raw_edf = mne.set_bipolar_reference(raw_edf, "Sync1", "Sync2", "SyncRR")
    sync_channel0 = raw_edf.copy().pick_channels(["SyncRR"])
    data, times = sync_channel0[:]  # Extract data
    data = data.transpose()
    data = zscore(data)  # normalize
    plt.plot(np.asarray(times), data)
    diff_data_raw = np.insert(np.diff(data, axis=0), 0, 0)
    diff_data = diff_data_raw.copy()
    threshold_moving = window_stdev(diff_data, 5000) * 3.5  # Calulate a moving threshold by
    diff_data[np.where((diff_data > -threshold_moving) & (diff_data < threshold_moving))] = 0  # Threshold
    diff_data[np.where(diff_data < 0)] = -diff_data[np.where(diff_data < 0)]  # Rectify

    plt.plot(np.asarray(times), diff_data)
    sync_times = detect_peaks(diff_data, mph=0.2)
    sync_times_sign = np.where(diff_data_raw[sync_times] > 0, 1, 0)
    rising = sync_times[sync_times_sign]
    falling = sync_times[~sync_times_sign]
    plt.plot(rising / sample_rate, np.ones(rising.shape), '*')
    plt.plot(falling / sample_rate, np.ones(falling.shape), '*')
    sync_times_secs = sync_times / sample_rate

    timing_data_matlab_raw = pd.read_excel(timing_filename)
    timing_data_matlab_raw["getSecsLogTime"] -= timing_data_matlab_raw["getSecsLogTime"][0]  # Start at zero

    timing_data_matlab = add_expected_transitions(timing_data_matlab_raw)

    max_expected_lag = 5  # Differneces between matlab and eeg should not be more than 5

    sync_sorted_idx = 0
    eeg_events = {}  # Dict to hold all sync times

    tolerance_dict = {'wordFind_BeforeInter_findTimes': 0.20,
                      'questionare_BeforeInter_dispTime': 0.15,
                      'questionare_BeforeInter_respTime': 0.15,
                      'wordFind_AfterInter_findTimes': 0.50,
                      'questionare_AfterInter_dispTime': 0.15,
                      'questionare_AfterInter_respTime': 0.15,
                      'emotionMemRecog_BeforeInter_recogRespTime': 0.1,
                      'emotionMemRecog_AfterInter_recogRespTime': 0.1}

    timing_data_matlab['label_generic'] = timing_data_matlab['label'].apply(lambda lab: lab[0:lab.rfind("_")])
    matlab_labels = timing_data_matlab['label_generic'].unique()
    for label in matlab_labels:
        matlab_events = timing_data_matlab.sync_time[label == timing_data_matlab['label_generic']]
        print("For", label, "attempting to find", len(matlab_events), "events")
        # plt.figure()
        # plt.plot(matlab_events, np.ones(len(matlab_events)), '*')
        # a = sync_times_secs[(sync_times_secs < (matlab_events.iloc[-1] + max_expected_lag)) & (sync_times_secs > (matlab_events.iloc[0] - max_expected_lag))]
        # plt.plot(a, np.ones(len(a)), '*')
        # plt.show()
        expected_diff = np.diff(matlab_events)
        if label in ['syncDataLight_BeforeInter_flipTime']:
            sync_idx, sync_lag = find_template_in_sequence(sync_times_secs, expected_diff, 0.06, 0,
                                                           sync_times_secs[-1] / 8)
            sync_times_secs = sync_times_secs[sync_idx[0]:] - sync_lag
            print('Found start at', sync_lag)
        else:
            if label in tolerance_dict:
                tolerance = tolerance_dict[label]
            else:
                tolerance = 0.08
            sync_idx, sync_lag = find_template_in_sequence(sync_times_secs, expected_diff, tolerance,
                                                           matlab_events.iloc[0] - max_expected_lag,
                                                           matlab_events.iloc[-1] + max_expected_lag)
            #Linearly increasing lag
            max_expected_lag = 5+40*(matlab_events.iloc[0]/max(timing_data_matlab.sync_time))

        if sync_idx is None:
            print('Did not find all points for', label)
            continue

        print('Updating lag to', max_expected_lag)
        print("Found", len(sync_idx), "events at", sync_lag)
        eeg_events[label] = sync_times_secs[sync_idx]
        sync_sorted_idx += len(sync_idx)

    plot_sync_times(sync_times_secs, eeg_events, timing_data_matlab)


def add_expected_transitions(sync_times_secs):
    sync_template = pd.read_excel("syncTemplate.xlsx")
    sync_times_with_expected_trans = pd.DataFrame(columns=('label', 'sync_time', 'trans'))
    idx = 0

    for idx_sync_times, og_sync_row in sync_times_secs.iterrows():
        lab_to_match = og_sync_row.label.replace('BeforeInter_', '')
        lab_to_match = lab_to_match.replace('AfterInter_', '')
        try:
            loc_in_template = np.flatnonzero(lab_to_match == sync_template.label)[0]
        except:
            print('Did not find', lab_to_match)
        if sync_template.label[loc_in_template - 1] == 'prepend':
            sync_times_with_expected_trans.loc[idx] = ['prepend', np.nan, sync_template.trans[loc_in_template - 1]]
            idx += 1

        if 'replace' == sync_template.notes[loc_in_template]:
            sync_times_with_expected_trans.loc[idx] = ['replace', np.nan, sync_template.trans[loc_in_template]]
            idx += 1
        else:
            sync_times_with_expected_trans.loc[idx] = [og_sync_row.label, og_sync_row.getSecsLogTime,
                                                   sync_template.trans[loc_in_template]]
            idx += 1

        if sync_template.label[loc_in_template + 1] == 'append':
            sync_times_with_expected_trans.loc[idx] = ['append', np.nan, sync_template.trans[loc_in_template + 1]]
            idx += 1

    colormap = {'Black': 1, 'White': 0}
    sync_times_with_expected_trans['trans_bool'] = sync_times_with_expected_trans.trans.map(colormap)
    trans_diffs = np.insert(np.diff(sync_times_with_expected_trans.trans_bool), 0, 1)
    sync_times_with_expected_trans = sync_times_with_expected_trans.loc[trans_diffs.nonzero()[0], :]
    return sync_times_with_expected_trans[
        (sync_times_with_expected_trans.label != 'append')
        & (sync_times_with_expected_trans.label != 'prepend')
        & (sync_times_with_expected_trans.label != 'replace')]


def plot_sync_times(sync_times_secs, eeg_events, timing_data_events):
    plt.figure()

    for label_type in timing_data_events['label_generic'].unique():
        data_to_plot_matlab = timing_data_events.sync_time[label_type == timing_data_events['label_generic']]
        p = plt.plot(data_to_plot_matlab, np.ones(len(data_to_plot_matlab)), '*', label=label_type)

        if label_type in eeg_events:
            data_to_plot_eeg = eeg_events[label_type]
            plt.plot(data_to_plot_eeg, np.ones(len(data_to_plot_eeg)), '^', label=label_type, color=p[0].get_color())

    plt.plot(sync_times_secs, np.ones(len(sync_times_secs)), '.')
    plt.show()


def find_template_in_sequence(data_seq, template, tolerance=0, start_looking=0, end_looking=None):
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
    for idx, point in enumerate(data_seq):
        matched_idxs = [idx]
        next_point = point
        found_pattern = True
        for interval in template:
            next_point += interval
            diff_to_target = np.abs(data_seq - next_point)
            closest_point_idx = np.argmin(diff_to_target)
            closest_point = np.min(diff_to_target)
            if closest_point > tolerance:
                #print('Error, closest point is:', closest_point, 'and only found', len(matched_idxs), 'matching points')
                found_pattern = False
                break
            next_point = data_seq[closest_point_idx]
            matched_idxs.append(closest_point_idx)
        if found_pattern:
            return np.array(matched_idxs) + num_low_values_to_drop, point

    warnings.warn("Pattern not found")
    return (None, None)


if __name__ == "__main__":
    edf_loc = "../Data/EEG/"
    behav_loc = "../Data/FeelData/EEGStudy1/Timestamps/"
    process_events(edf_loc + "PAI_2028.edf", behav_loc + "PAI_RawData_Subject2028_EEGSync_Timestamps.xlsx")
