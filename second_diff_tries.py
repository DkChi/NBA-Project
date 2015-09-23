# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 22:24:45 2015

@author: User
"""
import sys
if 'D:\Gal\Work' not in sys.path:
    sys.path.append('D:\Gal\Work')
import classes4NBA as c
import technical_functions_NBA as tf
import research_functions_NBA as rf
import numpy as np
import matplotlib.pyplot as mpl
from collections import Counter


def second_differtional4game(g, time_unit=30):
    ''''''
    SIZE = int(4*c.Q_LEN/time_unit)
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
    return avg_sec_diff[:SIZE]

def second_diffrentioal(team='SAS', time_unit=30,
                        s_date='10000101', f_date='21001231'):
    ''' '''
    SIZE = int(4*c.Q_LEN/time_unit)
    
    #Home Games
    games = tf.all_games_in_spec_dates(tf.all_games_per_team(team, away=False),
                                       s_date, f_date)
    sd_over_games = np.zeros(SIZE)
    for g in games:
        #print g.title()         
        sd_points = second_differtional4game(g, time_unit)
        for i in xrange(len(sd_points)):
            sd_over_games[i] += sd_points[i]
    
    #Away Games
    games = tf.all_games_in_spec_dates(tf.all_games_per_team(team, home=False),
                                       s_date, f_date)        
    for g in games:
        #print g.title()         
        sd_points = second_differtional4game(g, time_unit)
        for i in xrange(len(sd_points)):
            sd_over_games[i] -= sd_points[i]

    sd_over_games -= np.mean(sd_over_games)
            
    return sd_over_games
  

def avg_with_duplicates(times, data):
    ''''''
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
def compare2teams(team, team2, time_unit=120.0):    
    ''''''       
    #g = tf.create_game_from_file('20150520-CLEATL.txt')
    s_date='20141001'
    #sd_over_games = np.zeros(rf.Q_LEN*4/time_unit)
    #sd_point = second_differtional4game(g, time_unit)
    #rf.plot_diff(g)
    #mpl.plot(sd_point) 
    total_sd = tf.normalize(second_diffrentioal(team=team,time_unit=time_unit,
                                                s_date=s_date))                                             
    
    d = []                                            
    home_games = tf.all_games_in_spec_dates(tf.all_games_per_team(team=team2,
                                                                  away=False),
                                           s_date=s_date)
    for g in home_games:
        
        sd_points = second_differtional4game(g, time_unit)
        sd_points -= np.mean(sd_points)
        d.append(tf.distance(total_sd, tf.normalize(sd_points)))
    
        
    away_games = tf.all_games_in_spec_dates(tf.all_games_per_team(team=team2,
                                                                  home=False),
                                                                  s_date=s_date)
    for g in away_games:    
        sd_points = -1*np.array(second_differtional4game(g, time_unit))
        sd_points -= np.mean(sd_points)
        d.append(tf.distance(total_sd, tf.normalize(sd_points)))
        
    return d


def compare2all_teams(main_team, time_unit=120.0):
    '''Comparing the second difrential of the main team to all the teams'''
    dic = {}
    for team in tf.all_teams:
        d = compare2teams(main_team, team, time_unit=time_unit)
        #print team, 'median:', np.median(d), 'mean:', np.mean(d) # DEBUG
        dic[team] = np.mean(d)
    return dic
         

def compare_all2all(time_unit=120.0):
    '''Comparing the second difrential of each team to all the teams'''
    for team in tf.all_teams:
        dic = compare2all_teams(team,time_unit)
        print team , min(dic, key=dic.get)



def check_all(time_unit=120.0, s_date='20141001'):
    ''' '''
    total_sd = {}
    for team in tf.all_teams:
        total_sd[team] = tf.normalize(second_diffrentioal(team, time_unit, s_date=s_date))
    
    del team    
    
    for team in tf.all_teams:   
        d = Counter()
        home_games = tf.all_games_in_spec_dates(tf.all_games_per_team(team=team,
                                                                  away=False),
                                                                  s_date=s_date)
        for g in home_games:
            sd_points = second_differtional4game(g, time_unit)
            sd_points -= np.mean(sd_points)
            for team1 in tf.all_teams:
                d[team1] += (tf.distance(total_sd[team1], tf.normalize(sd_points)))
        
        away_games = tf.all_games_in_spec_dates(tf.all_games_per_team(team=team,
                                                                  home=False),
                                                                  s_date=s_date)
        del team1, g                                                          
        for g in away_games:    
            sd_points = -1*np.array(second_differtional4game(g, time_unit))
            sd_points -= np.mean(sd_points)
            for team1 in tf.all_teams:
                d[team1] += (tf.distance(total_sd[team1], tf.normalize(sd_points)))
            
        print team, min(d, key=d.get)


#check_all(240.)


#g = tf.create_game_from_file('20150520-CLEATL.txt')
#sd_point = second_differtional4game(g, time_unit=120.)
#rf.plot_diff(g)
#mpl.plot(sd_point)     
#mpl.plot(compare2teams(team, 'GSW',time_unit=time_unit),'.')
#mpl.hold(1)
path = r'D:\Gal\Work\sec_diff_'
tu = 120.0
for team in tf.all_teams:
    sd = second_diffrentioal(team,time_unit=tu)
    mpl.figure(figsize=(24,10))
    mpl.plot(tf.normalize(sd))
    mpl.title(team)
    mpl.xlim([0,24])
    mpl.xlabel('Time (in multiplies of 2 min.)')
    mpl.ylim([-1,1])                                         
    mpl.grid(axis='x')
    mpl.xticks(np.arange(1, 4)*6)  
    mpl.grid(axis='y')
    mpl.savefig(path+team+'.png')
    mpl.show()
    

                 