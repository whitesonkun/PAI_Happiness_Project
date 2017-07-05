## Imports
from process_emotiv import *
from process_feel import *
from eeg_analysis import *
from eeg_clean_channels import plot_potentially_bad_channels
import os.path

def load_eeg_data(sub_id: int, refresh_events: bool=False) -> object:
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
                                      init,
                                      refresh_events)

if __name__ == "__main__":

    behav_loc = "../Data/Behavioral/EEGStudy1/"

    # ideally import counterbalance of good and bad files
    sub_ids = [2019, 2022, 2024]
    sub_data = dotdict({})

    for sub_id in sub_ids:
        # sub_data.feel_data = load_feel_data(sub_id)
        sub_data.eeg, sub_data.events = load_eeg_data(sub_id)
        plot_potentially_bad_channels(sub_data.eeg)
        #extract_erps(sub_data.eeg.raw, sub_data.events.eeg_samples['syncDataLight_BeforeInter_flipTime'])

