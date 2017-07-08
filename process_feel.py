from scipy.io import loadmat
import datetime
from biosppy.signals import bvp
from parse import *
from yetti_utils import DotDict

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