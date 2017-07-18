class DotDict(dict):
    """
     a dictionary that supports dot notation
     as well as dictionary access notation
     usage: d = DotDict() or d = DotDict({'val1':'first'})
     set attributes: d.val2 = 'second' or d['val2'] = 'second'
     get attributes: d.val2 or d['val2']
     """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

import matplotlib.pyplot as plt
import numpy as np

def df_to_heatmap(data, cmap=plt.cm.Blues):
    # Plot it out
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data.values, cmap=cmap, alpha=0.8)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(data.columns.values.tolist(), minor=False)
    ax.set_yticklabels(data.index, minor=False)

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

from itertools import groupby

def ranges(lst):
    """Find the ranges of consecutive values in a list"""
    pos = (j - i for i, j in enumerate(lst))
    t = 0
    for i, els in groupby(pos):
        l = len(list(els))
        el = lst[t]
        t += l
        yield range(el, el + l)

def runs_of_ones_array(bits:np.ndarray):
    """Find start, end and length of consecutive ones in an array. handles bool or int/float types"""
    # make sure all runs of ones are well-bounded
    if bits.dtype == bool:
        bits = bits.astype(int)
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    run_lens = run_ends-run_starts
    return run_starts, run_ends, run_lens
