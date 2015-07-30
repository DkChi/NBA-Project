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
Q_LEN = 12
#p = tf.get_text('http://www.nba.com/games/20150514/HOULAC/gameinfo.html')

#im = tf.import_data(p)
print rf.shooting_stats_over_games(['Duncan'], 'SAS', f_date='20110701')
#for i in s :
#   print i, s[i], float(s[i][1])/s[i][0]*100
#DEBUG
#for i in xrange(len(im[0])) :
#   print i, im[0][i], im[1][i], im[2][i]
#DEBUG
#im = tf.read_from_file(r'D:\Gal\Work\Results\20150609-GSWCLE.txt')
#print rf.reb_after_ft(im[0], im[1], im[2])
#write_data(im[0], im[1], im[2],im[3])
#print starting5(p)
#data = tf.create_data(im[0], im[1], im[2])
#print time_on_court('Thompson',data)
#shots= []
#for d in data:
#    if type(d) is c.Shot :
#        shots.append(d)

#rf.plot_scores(im[0], im[1], im[2], im[3])
#rf.plot_diff(im[0], im[1], im[2])
#rf.shots4time(player='Anthony', team='NYK', s_date='20141001')


#tf.write_all_data(s_date='20150219', f_date='20150609')
#rf.starting_5_shots(team='ATL', starting_5=['Carroll', 'Millsap', 'Horford', 'Korver', 'Teague'], s_date='20141001')

#rf.shots4time('Curry', 'GSW','20141001')
#g = tf.create_Game('20150609-GSWCLE.txt')
#print g.get_date()
#print g.get_Home_Team()
#print g.get_Away_Team()
#print g.get_Home_Actions()
#print g.get_Away_Actions()
'''
actions_1, played_games_1 = rf.action_raster24('Curry', 'GSW', action=c.Shot)
actions_2, played_games_2 = rf.action_raster24('Thompson', 'GSW', action=c.Shot)


ah_1 = rf.actions_histogram(rf.get_timing_histogram(actions_1), 1)
ah_2 = rf.actions_histogram(rf.get_timing_histogram(actions_2), 1)


mpl.plot(ah_1, 'r')
mpl.plot(ah_2, 'b')
mpl.xlim([0, 26])
mpl.figure(3)
mpl.plot(tf.normalize(ah_1), 'r')
mpl.plot(tf.normalize(ah_2), 'b')
mpl.xlim([0, 26])

'''
'''
mpl.hold(1)
mpl.figure(1)
colors = ['r', 'gray', 'b', 'gray', 'gray', 'm', 'gray', 'g']
actions, played_games = [0]*8, [0]*8
s_date = '20141030'


actions[0], played_games[0] = rf.time_raster('Durant', 'OKC', s_date=s_date)
actions[1], played_games[1] = rf.time_raster('Westbrook', 'OKC', s_date=s_date)
actions[2], played_games[2] = rf.time_raster('Thompson', 'GSW', s_date=s_date)
actions[3], played_games[3] = rf.time_raster('Paul', 'LAC', s_date=s_date)
actions[4], played_games[4] = rf.time_raster('Harden', 'HOU', s_date=s_date)
actions[5], played_games[5] = rf.time_raster('Duncan', 'SAS', s_date=s_date)
actions[6], played_games[6] = rf.time_raster('Aldridge', 'POR', s_date=s_date)
actions[7], played_games[7] = rf.time_raster('Curry', 'GSW', s_date=s_date)


for i in xrange(0,8):
    mpl.plot(tf.normalize(tf.movingAverage(rf.actions_histogram(actions[i]), 3)), colors[i])

mpl.show()
'''
'''

players = ['James', 'Paul', 'Curry', 'Thompson', 'Duncan']
teams = ['MIA', 'LAC', 'GSW', 'GSW', 'SAS']
colors = ['r', 'b', 'm', 'k', 'g']
actions = [0]*len(players)
played_games = [0]*len(players)
s_date = '20100901'

mpl.hold(1)
mpl.figure(1)
for i in xrange(len(players)):
    actions[i], played_games[i] = rf.time_raster(players[i], teams[i], s_date)
    mpl.plot(tf.normalize(tf.movingAverage(rf.actions_histogram(actions[i]), 3)), colors[i])

mpl.figure(2)
for i in xrange(len(players)):
    actions[i], played_games[i] = rf.actions_raster([players[i]], teams[i], s_date, action=c.Assist)
    mpl.plot(tf.normalize(tf.movingAverage(rf.actions_histogram(actions[i]), 3)), colors[i])

mpl.show()
'''
'''
LIMIT = 25
atlantic_teams = ['BOS', 'BKN', 'NYK', 'PHI', 'TOR']
central_teams = ['CHI', 'IND', 'DET', 'CLE', 'MIL']
southeast_teams = ['ATL', 'CHA', 'MIA', 'ORL', 'WAS']
southwest_teams = ['DAL', 'SAS', 'HOU', 'MEM', 'NOP']
pacific_teams = ['LAL', 'LAC', 'GSW', 'PHX', 'SAC']
northwest_teams = ['DEN', 'MIN', 'OKC', 'POR', 'UTA']
all_teams = atlantic_teams + central_teams + southeast_teams + southwest_teams + pacific_teams + northwest_teams

for team in all_teams:
    print team, rf.reb_after_tf_per_team(team)
'''
'''
POSITIONS = ['pg', 'sg', 'sf', 'pf', 'c']
BASE_LINK_ESPN = r'http://espn.go.com/nba/players/_/position/'
players_by_kind = []
for i in xrange(len(POSITIONS)):
    players_by_kind.append(tf.bring_players_from_espn(tf.get_text(BASE_LINK_ESPN+POSITIONS[i])))


path = 'D:\Gal\Work\Results'
f = open(path+'\\'+'players_good_shots_checking.txt', 'wb')
for players in players_by_kind:

    for i in players:
        print i
        comma = i.find(',')
        last_name = i[:comma]

        for t in all_teams:
            shots, games = rf.actions_raster(last_name, t, s_date='20131031', f_date='21040801') # has a problem because the function returns times and not shots
            print len(shots)
            if len(shots) > 100:
                print i, t
                f.write(i + ',' + t + ',' + rf.good_shots_checking_given_shots(shots, i)[0]+','+rf.good_shots_checking_given_shots(shots, i)[1] + '\n')

f.close()
'''