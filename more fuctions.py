import re
import numpy as np
import matplotlib.pyplot as mpl

import os
import sys
if 'D:\Gal\Work' not in sys.path:
    sys.path.append('D:\Gal\Work')
import classes4NBA as c
import technical_functions_NBA as tf
import research_functions_NBA as rf
__author__ = 'GalBien'

mpl.figure(1)
actions_1, played_games_1 = rf.actions_raster('Curry', 'GSW', s_date='20141001')
b, c = [], []
for a in actions_1:
    if a.get_action().get_kind() == '3pt Shot':
        b.append(a)
        if a.get_action().is_made():
            c.append(a)

ah_1 = rf.actions_histogram(rf.get_timing_histogram(actions_1), 1)
ah_2 = rf.actions_histogram(rf.get_timing_histogram(b), 1)
ah_3 = rf.actions_histogram(rf.get_timing_histogram(c), 1)

shot_percentage = []
for i in xrange(np.min([len(ah_3), len(ah_2)])):
    shot_percentage.append(float(ah_3[i]/ah_2[i])*100)


mpl.plot(ah_1, 'r')
mpl.plot(ah_2, 'b')
mpl.plot(ah_3, 'g')
mpl.xlabel('Time (in seconds)')
mpl.xlim([
mpl.grid(axis='x')


mpl.figure(2)
mpl.plot(tf.normalize(ah_1), 'r')
mpl.plot(tf.normalize(ah_2), 'b')
mpl.plot(tf.normalize(ah_3), 'g')
mpl.xlabel('Time (in seconds)')
mpl.xlim([0, 25])
mpl.grid(axis='x')

mpl.figure(3)
mpl.plot(shot_percentage)
mpl.xlabel('Time (in seconds)')
mpl.xlim([0, 25])
mpl.grid(axis='x')

mpl.show()