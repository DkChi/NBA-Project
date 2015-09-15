# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:24:45 2015

@author: User
"""
if 'D:\Gal\Work' not in sys.path:
    sys.path.append('D:\Gal\Work')
import classes4NBA as c
import technical_functions_NBA as tf
import research_functions_NBA as rf
import numpy as np
import matplotlib.pyplot as mpl

def second_differtional4game(g, time_unit=30):
    times, point_diff = rf.diff(g)
    sec_diff_point = []
    for t in times:
        min_time, max_time = float(t)-time_unit, float(t)+time_unit
        i, start, end = 0, False, False
        while i < len(times):
            if min_time < float(times[i]) and not start:
                start_diff = point_diff[i]
                start = True
            elif max_time < float(times[i]) and not end:
                end_diff = point_diff[i]
                end = True
            i += 1
        if not start:
            start_diff = 0
        if not end:
            end_diff = point_diff[-1]
            
        sec_diff_point.append(end_diff-start_diff)
        
    new_times = np.floor(np.array(times)/time_unit)
    avg_sec_diff = avg_with_duplicates(new_times, sec_diff_point)   
    return avg_sec_diff

def second_diffrentioal(team='SAS', time_unit=30,
                        s_date='10000101', f_date='21001231'):
    ''' '''
    games = tf.all_games_in_spec_dates(tf.all_games_per_team(team, away=False),
                                       s_date, f_date)
    sd_over_games = np.zeros(rf.Q_LEN*6/time_unit)
    for g in games:
        #print g.title()         
        sd_points = second_differtional4game(g, time_unit)
        for i in xrange(len(sd_points)):
            sd_over_games[i] += sd_points[i]
    return sd_over_games
  

def avg_with_duplicates(times, data):
    last = 0 
    cnt = 0
    result = [0]
    for i, t in enumerate(times):
        if last == t:
            result[-1] += data[i]
            cnt += 1
        else:
            result[-1] /= cnt
            result.append(data[i])
            cnt = 1
            last = t
                
    result[-1] /= cnt
    return result
        
#------ MAIN ----------
            
#g = tf.create_game_from_file('20150520-CLEATL.txt')
time_unit = 120.0
#sd_over_games = np.zeros(rf.Q_LEN*4/time_unit)
#sd_point = second_differtional4game(g, time_unit)
#rf.plot_diff(g)
#mpl.plot(sd_point) 
mpl.hold(1) 
mpl.xlim([0,rf.Q_LEN*4/time_unit])                                         
mpl.grid(axis='x')
mpl.xticks(np.arange(1, 4)*rf.Q_LEN/time_unit)  
mpl.grid(axis='y')
mpl.yticks([0])
mpl.plot(tf.normalize(second_diffrentioal(team='CLE',time_unit=120,
                                           s_date='20141001')))


mpl.show()
     
         
                 
                 