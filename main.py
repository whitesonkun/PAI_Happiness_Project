## main function ##

## Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne


## Read EEG
data_loc = "../Data/EEG/"
raw_edf = mne.io.read_raw_edf(data_loc + "PAI_2013.edf", stim_channel=None, exclude={})
raw_edf.plot()
plt.show()

#
	