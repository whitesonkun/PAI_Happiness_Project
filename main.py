## Imports
from process_emotiv import *
from process_feel import *
from eeg_analysis import *
from eeg_preprocessing import interactive_channel_cleaner
from yetti_utils import *
import os.path
import sys

def load_eeg_data(sub_id: int) -> object:
    edf_loc = "../Data/EEG/"
    timestamp_loc = "../Data/FeelData/EEGStudy1/Timestamps/"
    working_data_loc = './working_data/'

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

    return sync_matlab_and_eeg_events(edf_loc + "PAI_" + str(sub_id) + ".edf",
                                      timestamp_loc + "PAI_RawData_Subject" + str(sub_id) + "_EEGSync_Timestamps.xlsx",
                                      working_data_loc + "PAI_" + str(sub_id) + "_Events.xlsx",
                                      init)

def save_sub_data(sub_data):
    """Load exisitn subject data for a subject if it exisits"""

    file_name = "working_data/PAI_wd_sub"+str(sub_data.meta.sub_id)
    sub_data.events.to_csv(file_name+'_events.csv')
    try:
        os.remove(file_name + '_raw.fif')
    except FileNotFoundError:
        pass

    try:
        os.remove(file_name + '_impedance.fif')
    except FileNotFoundError:
        pass

    with open(file_name+'_meta.txt', 'w') as f:
        json.dump(sub_data.meta, f, ensure_ascii=False)

    sub_data.eeg.raw.save(file_name + '_raw.fif', overwrite=True)
    sub_data.eeg.impedance.save(file_name + '_impedance_raw.fif', overwrite=True)


def load_sub_data(sub_id):
    sub_data = init_new_subject(sub_id)
    file_name = "working_data/PAI_wd_sub" + str(sub_data.meta.sub_id)
    try:
        sub_data.events = pd.read_csv(file_name+'_events.csv')
        sub_data.eeg.raw = mne.io.read_raw_fif(file_name + '_raw.fif')
        sub_data.eeg.raw.load_data()
        sub_data.eeg.impedance = mne.io.read_raw_fif(file_name + '_impedance_raw.fif')
        sub_data.eeg.impedance.load_data()
        with open(file_name+'_meta.txt') as r:
            sub_data.meta = DotDict(json.load(r))
    except FileNotFoundError:
        return False

    return sub_data


def init_new_subject(sub_id):
    """Created the object to hold a subjects data"""
    sub_data = DotDict({'meta': {'sub_id': sub_id, 'artifacting_params':{}}, 'eeg': {}})
    return sub_data

if __name__ == "__main__":
    refresh_data = False

    behav_loc = "../Data/Behavioral/EEGStudy1/"

    # ideally import counterbalance of good and bad files
    sub_ids = [2019, 2024] #Need to work on 2022

    for sub_id in sub_ids:
        sub_data = init_new_subject(sub_id) if refresh_data else (load_sub_data(sub_id) or init_new_subject(sub_id))
        if ('eeg' not in sub_data) or ('events' not in sub_data):
            sub_data.eeg, sub_data.events = load_eeg_data(sub_id)
            add_event_channel_to_eeg(sub_data.eeg.raw, sub_data.events)

        #Preproccessing
        sub_data.meta.artifacting_params = interactive_channel_cleaner(sub_data.eeg.raw, sub_data.eeg.impedance)
        if sub_data.meta.artifacting_params is None:
            break
        save_sub_data(sub_data)

        #extract_erps(sub_data.eeg.raw, sub_data.events.eeg_samples['syncDataLight_BeforeInter_flipTime'])

    print('Program Done. Exiting')