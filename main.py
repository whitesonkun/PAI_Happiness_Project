## main function ##

## Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from detect_peaks import detect_peaks
import mne
from scipy import signal
from scipy.stats import zscore
from scipy.ndimage.filters import uniform_filter
from scipy.spatial.distance import euclidean
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

    marker_seqs = {'exp_start': 0.1*np.ones(19),
                   'task_start': [0.05, 0.08, 0.1, 0.1]}

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
    diff_data = np.insert(np.diff(data, axis=0), 0, 0)
    threshold_moving = window_stdev(diff_data, 5000) * 3  # Calulate a moving threshold by
    diff_data[np.where((diff_data > -threshold_moving) & (diff_data < threshold_moving))] = 0  # Threshold
    diff_data[np.where(diff_data < 0)] = -diff_data[np.where(diff_data < 0)]  # Rectify

    plt.plot(np.asarray(times), diff_data)
    sync_times = detect_peaks(diff_data, mph=0.2)
    plt.plot(sync_times / sample_rate, np.ones(sync_times.shape), '*')
    sync_times_secs = sync_times / sample_rate
    #plt.show()

    timing_data = pd.read_excel(timing_filename)
    timing_data["getSecsLogTime"] -= timing_data["getSecsLogTime"][0]

    eeg_events = {} #Dict to hold all sync times
    sync_idx, sync_lag = find_template_in_sequence(sync_times_secs, marker_seqs['exp_start'], 0.05, 0, sync_times_secs[-1]/8)
    print(sync_idx, sync_lag)
    sync_times_secs = sync_times_secs[sync_idx[0]:] - sync_lag
    eeg_events['syncDataLight_BeforeInter_flipTime'] = sync_times_secs[0:len(sync_idx)]
    eeg_events['toIgnore'] = sync_times_secs[len(sync_idx)+1]
    print(sync_times_secs[0:len(sync_idx)])

    #TODO: Now go through the timestamp file and extract unique labels, and autodiff this and send into function

    plot_sync_times(sync_times_secs, eeg_events, timing_data)
    # for task in num_tasks:
    #     sync_lag, sync_idx = find_sequence_lag_fuzzy_interval(sync_times_secs, 10, 0.1, 0.05)
    #     sync_times_secs = sync_times_secs[sync_idx:] - sync_lag

def plot_sync_times(sync_times_secs, eeg_events, timing_data_events):
    plt.figure()

    timing_data_events['label_generic'] = timing_data_events['label'].apply(lambda label: label[0:label.rfind("_")])
    label_types = timing_data_events['label_generic'].unique()
    for label_type in label_types:
        data_to_plot_matlab = timing_data_events.getSecsLogTime[label_type == timing_data_events['label_generic']]
        p = plt.plot(data_to_plot_matlab, np.ones(len(data_to_plot_matlab)), '*', label=label_type)

        if label_type in eeg_events:
            data_to_plot_eeg = eeg_events[label_type]
            plt.plot(data_to_plot_eeg, np.ones(len(data_to_plot_eeg)), '^', label=label_type, color=p[0].get_color())

    plt.plot(sync_times_secs, np.ones(len(sync_times_secs)), '.')
    plt.show()

def find_template_in_sequence(data_seq, template, tolerance=0, start_looking=0, end_looking=None):
    # FIXME: this function should take a pattern in seconds, not difference of seconds. Diff can be calculated internally
    """ 
    Find a specific pattern of numbers in an sorted array. Small deviations from the pattern are tolerated.
    
    Args:
        data_seq (ndarray): The array to find the matching pattern in
        template (ndarray): pattern of differences between elements in array
        tolerance (int): how much each point in the template may be off by and a match still found
        start_looking (int): value to begin searching at
        end_looking (int): value to end searching at 

    Returns:
        int: the value where the template is first found
        int: the index where the template is first found
    """
    if end_looking is None:
        end_looking = max(data_seq)

    vals_ok_low = data_seq <= end_looking
    vals_ok_high = data_seq >= start_looking
    num_low_values_to_drop = sum(vals_ok_high == False)
    data_seq = data_seq[vals_ok_low & vals_ok_high]
    for idx, point in enumerate(data_seq):
        matched_idxs = []
        matched_idxs.append(idx)
        next_point = point
        found_pattern = True
        for interval in template:
            next_point += interval
            a = (data_seq < next_point + tolerance) & (data_seq > next_point - tolerance)
            matched_idx = np.where(a)
            if not a.any():
                found_pattern = False
                break
            matched_idxs.append(matched_idx[0])
        if found_pattern:
            return np.array(matched_idxs)+num_low_values_to_drop, point

    warnings.warn("Pattern not found")
    return

# def find_sequence_lag_overlap(data_seq, template_seq, sample_rate, std_tolerance_in_samples):
#     # First we add gaussian around each point gaussian array
#     data_seq_g = np.zeros(int(max(data_seq) * sample_rate) + sample_rate)
#     template_seq_g = np.zeros(int(max(template_seq) * sample_rate+0.1*sample_rate))
#     width_gauss = 6 * std_tolerance_in_samples
#     gauss = signal.gaussian(width_gauss, std_tolerance_in_samples)
#
#     for point in template_seq:
#         min_loc = int(point * sample_rate - width_gauss / 2)
#         max_loc = int(point * sample_rate + width_gauss / 2)
#         gauss_min = 0
#         gauss_max = len(gauss)
#         if min_loc < 0:
#             gauss_min = -min_loc
#             min_loc = 0
#         if max_loc > len(template_seq_g):
#             gauss_max = len(template_seq_g) - max_loc
#             max_loc = template_seq_g
#
#         template_seq_g[min_loc:max_loc] += gauss[gauss_min:gauss_max]
#
#     for point in data_seq:
#         min_loc = int(point * sample_rate - width_gauss / 2)
#         max_loc = int(point * sample_rate + width_gauss / 2)
#         gauss_min = 0
#         gauss_max = len(gauss)
#         if min_loc < 0:
#             gauss_min = -min_loc+1
#             min_loc = 0
#         if max_loc > len(data_seq_g):
#             gauss_max = len(data_seq_g) - max_loc
#             max_loc = data_seq_g
#         data_seq_g[min_loc:max_loc] += gauss[gauss_min:gauss_max]
#
#     #xcorr = np.correlate(data_seq_g, template_seq_g, mode='full')
#     crossproduct = []
#     SSE_old = 0
#     for sample in range(0, len(data_seq_g) - len(template_seq_g)):
#         part_data_seq = data_seq_g[sample:(sample + len(template_seq_g))]
#         dist = fastdtw(part_data_seq, template_seq_g, dist=euclidean)
#         SSE = np.sum(np.power(part_data_seq - template_seq_g, 2))
#         crossproduct.append(dist)
#
#         if SSE > 53:
#             print('woot')
#             # plt.plot(part_data_seq)
#             # plt.plot(template_seq_g)
#             # plt.plot(np.power(part_data_seq - template_seq_g, 2))
#             # plt.show()
#
#         SSE_old = SSE
#
#
#     ax1 = plt.subplot(4, 1, 1)
#     plt.plot(data_seq_g)
#
#     plt.subplot(4, 1, 2, sharex=ax1)
#     plt.plot(template_seq_g)
#
#     plt.subplot(4, 1, 3, sharex=ax1)
#     plt.plot(crossproduct)
#
#     lag_in_samples = np.argmin(crossproduct)
#     lagged_template = np.concatenate([np.zeros(lag_in_samples), template_seq_g[0:-lag_in_samples]])
#
#     plt.subplot(4, 1, 4, sharex=ax1)
#     plt.plot(data_seq_g)
#     plt.plot(lagged_template)
#     plt.plot(lag_in_samples,1, '*')
#
#     plt.show()
#     return np.argmin(crossproduct) / sample_rate

if __name__ == "__main__":
    edf_loc = "../Data/EEG/"
    behav_loc = "../Data/FeelData/EEGStudy1/Timestamps/"
    edf_with_loc = process_events(edf_loc+"PAI_2029.edf", behav_loc+"PAI_RawData_Subject2029_EEGSync_Timestamps.xlsx")
