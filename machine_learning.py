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
import matplotlib.pyplot as mpl

def read_shotchartdetail(player_id, season='2014-15'):
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


def check_machine_for_all_players():
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
    players = pd.DataFrame(response.json()['resultSets'][0]['rowSet'], columns=headers)
    players_dic = {}
    for player, player_id, fga, gp in zip(players.PLAYER_NAME, players.PLAYER_ID, players.FGA, players.GP):
        if gp*fga > 100:
            players_dic[player] = player_id
            
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
    print np.array(k_scores).mean(),

shot_chart_url2 = 'http://stats.nba.com/stats/playerdashptshotlog?DateFrom=&'\
                  'DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&'\
                  'Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID={0}&'\
                  'Season={1}&SeasonSegment=&SeasonType=Regular+Season&'\
                  'TeamID=0&VsConference=&VsDivision='

player_id = '1495'
season = '2014-15'  
shots1 = read_shotchartdetail(player_id, season)
response = requests.get(shot_chart_url2.format(player_id, season))
headers = response.json()['resultSets'][0]['headers']
shots2 = response.json()['resultSets'][0]['rowSet']
feature_cols =  ['PERIOD', 'LOC_X', 'LOC_Y','SHOT_MADE_FLAG']

shot_df1 = pd.DataFrame(shots1[1], columns=shots1[0])
shot_df1.sort(['GAME_ID', 'GAME_EVENT_ID'], ascending=[False,True] , inplace=True)
shot_df1 = pd.DataFrame(add_text_col(shot_df1, feature_cols, 'ACTION_TYPE'),
                        columns=feature_cols)
                       
shot_df1_index = pd.Series(xrange(len(shot_df1)))
#shot_df1 = pd.concat([shot_df1, shot_df1_index], axis=1)                        

shot_df2 = pd.DataFrame(shots2, columns=headers)
shot_df2.sort(['GAME_ID','SHOT_NUMBER'], ascending=[False,True], inplace=True)
shot_df2_index = pd.Series(xrange(len(shot_df2)))
#shot_df2 = pd.concat([shot_df2, shot_df2_index], axis=1)

shot_df = pd.concat([shot_df1, shot_df2], axis =1, join='inner')



feature_cols =  ['PERIOD', 'LOC_X', 'LOC_Y','ACTION_TYPE',
                 'TOUCH_TIME', 'DRIBBLES', 'CLOSE_DEF_DIST']
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
