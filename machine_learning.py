# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 13:29:40 2015

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:01:34 2015

@author: User
"""
import requests
import sys
import numpy as np
import pandas as pd
if 'C:\Anaconda\Lib\site-packages' not in sys.path :
    sys.path.append('C:\Anaconda\Lib\site-packages')
#from mpl_toolkits.mplot3d import Axes3d
import matplotlib.pyplot as mpl


def read_shotchartdetail(player_id='2544', season='2014-15'):
    shot_chart_url = 'http://stats.nba.com/stats/shotchartdetail?'\
                     'Period=0&VsConference=&LeagueID=00&LastNGames=0&TeamID=0&'\
                     'Position=&Location=&Outcome=&ContextMeasure=FGA&DateFrom=&'\
                     'StartPeriod=&DateTo=&OpponentTeamID=0&ContextFilter=&'\
                     'RangeType=&Season={1}&AheadBehind=&'\
                     'PlayerID={0}&EndRange=&VsDivision=&'\
                     'PointDiff=&RookieYear=&GameSegment=&Month=0&ClutchTime=&'\
                     'StartRange=&EndPeriod=&SeasonType=Regular+Season&SeasonSegment=&'\
                     'GameID='
                                         
    response = requests.get(shot_chart_url.format(player_id, season))
    headers = response.json()['resultSets'][0]['headers']
    return headers, response.json()['resultSets'][0]['rowSet']
  
  
def add_text_col(shot_df, feature_cols, new_col):
    i = 0
    shot_dic = {}
    new_col_list = []
    for shot in shot_df[new_col]:
        if shot not in shot_dic:
            shot_dic[shot]= i
            i += 1
        new_col_list.append(shot_dic[shot])       
    feature_cols.append(new_col)
    x = np.array(shot_df[feature_cols])
    for i in xrange(len(x)):
        x[i][-1] = new_col_list[i]     
    return x

def all_valid_palyers():
    all_players_url = 'http://stats.nba.com/stats/leaguedashplayerstats?College=&'\
                      'Conference=&Country=&DateFrom=&DateTo=&Division=&'\
                      'DraftPick=&DraftYear=&GameScope=&GameSegment=&'\
                      'Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&'\
                      'Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&'\
                      'PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&'\
                      'PlusMinus=N&Rank=N&Season={0}&SeasonSegment=&'\
                      'SeasonType=Regular+Season&ShotClockRange=&StarterBench=&'\
                      'TeamID=0&VsConference=&VsDivision=&Weight='.format('2014-15')
                    
    response = requests.get(all_players_url)
    headers = response.json()['resultSets'][0]['headers']
    players = pd.DataFrame(response.json()['resultSets'][0]['rowSet'],
                           columns=headers)
    players_dic = {}
    for player, player_id, fga, gp in zip(players.PLAYER_NAME,
                                          players.PLAYER_ID, 
                                          players.FGA,
                                          players.GP):
        if gp*fga > 100:
            players_dic[player] = player_id
            
    return players_dic


def machine_for_x_and_y():
    players_dic = all_valid_palyers()
            
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=50) 
    k_scores = []
    for player, player_id in players_dic.iteritems():
        feature_cols =  ['PERIOD', 'LOC_X', 'LOC_Y']
        print player, player_id
        shots = read_shotchartdetail(player_id,'2014-15')
        shot_df = pd.DataFrame(shots[1], columns=shots[0])
        x = add_text_col(shot_df, feature_cols, 'ACTION_TYPE')
        y = shot_df.SHOT_MADE_FLAG
        scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    
    mpl.plot(k_scores, '.')
    print np.array(k_scores).mean()

def machine_for_distance():
    players_dic = all_valid_palyers()
            
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=50) 
    k_scores = []
    for player, player_id in players_dic.iteritems():
        feature_cols =  ['TOUCH_TIME', 'DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST', 'PERIOD']
        headers, shots = read_playerdashptshotlog(player_id,'2014-15')
        shot_df = pd.DataFrame(shots, columns=headers)
        x = shot_df[feature_cols]
        y = shot_df.FGM
        scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
        print player, player_id, scores.mean() # DEBUG
        k_scores.append(scores.mean())
    
    mpl.plot(k_scores, '.')
    print np.array(k_scores).mean()


def read_playerdashptshotlog(player_id='2544',season='2014-15'):
    shot_chart_url = 'http://stats.nba.com/stats/playerdashptshotlog?DateFrom=&'\
                  'DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&'\
                  'Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID={0}&'\
                  'Season={1}&SeasonSegment=&SeasonType=Regular+Season&'\
                  'TeamID=0&VsConference=&VsDivision='
    response = requests.get(shot_chart_url.format(player_id, season))
    headers = response.json()['resultSets'][0]['headers']
    return headers, response.json()['resultSets'][0]['rowSet']
 
   
def learning_by_dist(player_id='1717',season='2014-15'):
    headers, shots = read_playerdashptshotlog(player_id, season)
    feature_cols =  ['TOUCH_TIME', 'DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST', 'PERIOD']
    shot_df = pd.DataFrame(shots, columns=headers)
    shot_clock = fixing_the_shot_clock(shot_df)
    
    shot_df = pd.concat([pd.Series(shot_clock),
                         pd.Series(shot_clock,name='NEW_SHOT_CLOCK')], axis=1)
    feature_cols =  ['TOUCH_TIME', 'DRIBBLES',
                     'SHOT_DIST', 'CLOSE_DEF_DIST',
                     'PERIOD','NEW_SHOT_CLOCK']
    x = shot_df[feature_cols]
    y = shot_df.FGM
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import cross_val_score
    k_range = range(1, 100)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    mpl.plot(k_range, k_scores)
    mpl.xlabel('Value of K for KNN')
    mpl.ylabel('Cross-Validated Accuracy')
    mpl.show()

    

def fixing_the_shot_clock(shot_df):
    shot_clock = list(shot_df.SHOT_CLOCK)
    new_shot_clock = []
    for s in shot_clock:
        if not np.isnan(s):
            new_shot_clock.append(s)
        else:
            new_shot_clock.append(-1)
    return new_shot_clock
    
    
    
#-----------------------------------------------------------------------------


#learning_by_dist()
#machine_for_x_and_y()
player_id ='202738'

headers, shots = read_shotchartdetail(player_id)
shot_df_x_y = pd.DataFrame(shots, columns=headers)

headers1, shots1 = read_playerdashptshotlog(player_id)
shot_df_dist = pd.DataFrame(shots1, columns=headers1)

shot_df_x_y.sort(columns=['GAME_ID','GAME_EVENT_ID'],inplace=True)


shot_df_dist.sort(columns=['GAME_ID','SHOT_NUMBER'],inplace=True)
array_dist = np.array(shot_df_dist)
shot_df = pd.concat([shot_df_x_y,pd.DataFrame(array_dist,columns=headers1)],
                     axis=1,join='inner')

print shot_df.GAME_ID
print shot_df.FGM-shot_df.SHOT_MADE_FLAG

 