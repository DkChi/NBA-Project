from collections import Counter
import numpy as np
import matplotlib.pyplot as mpl
import os
import sys
if 'D:\Gal\Work' not in sys.path :
    sys.path.append('D:\Gal\Work')
import classes4NBA as c
import technical_functions_NBA as tf

Q_LEN = 12*60
OT_LEN = 5*60


def quarter_time(q=0):
    ''' Returns the time when the quarter ends'''
    if q<=4:
        return q*Q_LEN
    else :
        return 4*Q_LEN+(q-4)*OT_LEN


def time_on_court(player, actions):
    ''' Returns array of how much time the player has been on court'''
    result = []
    q, t_in, flag = 1, -1, False

    for a in actions:
        t = float(a.time)
        flag = False

        if t > quarter_time(q):
            if t_in >= 0:
                #print 1, t_in, t # DEBUG
                if len(result) > 0:
                    if not t_in < result[-1][1]:
                        result.append([t_in, quarter_time(q)])
                else:
                    result.append([t_in, quarter_time(q)])
            t_in = -1
            q += 1

        if a.player == player and t_in < 0:  # the player came in between the quarters
            t_in = quarter_time(q-1)

        if type(a) is c.Sub:
            if a.player == player:  # the player came out
                #print 2, t_in, t, a # DEBUG
                if t_in < 0:
                    if len(result) > 0:
                        if not t_in < result[-1][1]:
                            result.append([quarter_time(q-1), t])
                    else:
                        result.append([quarter_time(q-1), t])

                else:
                    if len(result) > 0:
                        if not t_in < result[-1][1]:
                            result.append([t_in, t])
                    else:
                        result.append([t_in, t])
                t_in = -1

            elif a.player_in == player:  # the player came in
                if t_in >= 0:
                    if len(result) > 0:
                        if not t_in < result[-1][1]:
                            result.append([t_in, quarter_time(q-1)])
                    else:
                        result.append([t_in, quarter_time(q-1)])
                t_in = t

    if t_in > 0:
        if len(result) > 0:
            if not t_in < result[-1][1]:
                result.append([t_in, quarter_time(q)])
        else:
            result.append([t_in, quarter_time(q)])

    return result


def worth_points(this_shot):
    ''' '''
    if not this_shot.made:
        return 0  
        
    elif type(this_shot) is c.FreeThrow:
        return 1
        
    elif not this_shot.kind=='3pt Shot':
        return 2
        
    else:            
        return 3
    

def action_when_on_court(action, player, actions):
    ''' '''
    times = time_on_court(player, actions)
    for t in times:
        gt = float(action.time)
        if gt>np.array(t[0]) and gt<np.array(t[1]):
            return True
    return False        

def update_current_score(cnt, shot):
    cnt['score'] += worth_points(shot)
    cnt['index'] += 1
    return cnt


def scores(game):
    ''' 
    This function return the graph of the score in the game base only on the shots
    '''
    home_shots = tf.leave_these_actions(game.get_Home_Actions(), [c.Shot, c.FreeThrow])
    away_shots = tf.leave_these_actions(game.get_Away_Actions(), [c.Shot, c.FreeThrow])
    times, home_scores, away_scores = [],[],[]
    cnt_home, cnt_away = Counter(), Counter()

    for i in xrange (len(away_shots)+len(home_shots)):
        if cnt_home['index'] < len(home_shots) and cnt_away['index'] < len(away_shots):
            
            if float(away_shots[cnt_away['index']].time) > float(home_shots[cnt_home['index']].time) :
                this_shot = home_shots[cnt_home['index']]
                cnt_home = update_current_score(cnt_home, this_shot)
                
            else:
                this_shot = away_shots[cnt_away['index']]
                cnt_away = update_current_score(cnt_away, this_shot)
                
        elif cnt_away['index'] == len(away_shots):
            this_shot = home_shots[cnt_home['index']]
            cnt_home = update_current_score(cnt_home, this_shot)
                
        else:
            this_shot = away_shots[cnt_away['index']]
            cnt_away = update_current_score(cnt_away, this_shot)
                    
        #print this_shot.time,hp,ap, type(this_shot)
        times.append(this_shot.time)
        home_scores.append(cnt_home['score'])
        away_scores.append(cnt_away['score'])

    return times, home_scores, away_scores

     
def diff(g):
    ''' returns the point difference between the home team and the visitor team'''
    t, h, a = scores(g)
    return t, np.array(h)-np.array(a)

     
def plot_diff(g):
    '''plots the point difference between the home team and the visitor team '''
      
    diff_points = diff(g)
    mpl.plot(diff_points[0], diff_points[1], 'b')
    
    limits = []  
    if min(diff_points[1])<0 :
        limits.append( min(diff_points[1]))
    limits.append(0)
    if max(diff_points[1]) > 0:
        limits.append( max(diff_points[1]))
        
    mpl.xlabel('Time (in seconds)')
    mpl.ylabel('Point Differntial')
    mpl.grid(axis='x') 
    mpl.xticks(np.arange(1, 5)*Q_LEN)
    mpl.yticks(limits)
    mpl.title(g.title())
    mpl.show()


def plot_scores(g):
    ''' '''
    times, home_scores, away_scores = scores(g)
    mpl.hold(False)
    mpl.plot(times, home_scores, 'r')
    mpl.hold(True)
    mpl.plot(times, away_scores, 'b')
    mpl.xlabel('Time (in seconds)')
    mpl.ylabel('Points')
    mpl.grid(axis='x') 
    mpl.xticks(np.arange(1, 5)*Q_LEN)
    mpl.title(g.title())
    mpl.show()


def shooting_stats(shots, players=[]):
    '''Returns a dictionary of the shooting stats of spesific players '''
    dic = {}
    for s in shots:
        for p in players:
            if s.player == p:
                shot_kind = s.kind
                if not dic.has_key(shot_kind):
                    dic[shot_kind] = [0, 0]
                
                dic[shot_kind][0] += 1
                if s.made:
                    dic[s.kind][1] += 1
    return dic  


def shooting_stats_over_games(players=['Felton'], team='NYK', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    '''The function of dates nedd to added '''
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    dic = {}    
    for actions in actions_per_game:
        for a in actions:
            for p in players:
                if a.player == p and type(a) is c.Shot:
                    shot_kind = a.kind
                    if not dic.has_key(shot_kind):
                        dic[shot_kind] = [0, 0]
                    
                    dic[shot_kind][0] += 1
                    if a.made:
                        dic[a.kind][1] += 1
    return dic  


def free_throws_stats(freeThrows, players=[]):
    '''Returns the shooting stats of free throws for spesific players  '''
    attempted, made = 0, 0
    for ft in freeThrows:
        for p in players:
            if ft.player == p:
                attempted += 1
                if ft.made:
                    made += 1
    return made, attempted


def shooting_other(shooting_player='Parker', over_player='Duncan', team='SAS', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' This function calculates the shots of spesific player when other player is on the court '''    
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    dic = {}    
    for actions in actions_per_game:
        for a in actions:
            if a.player == shooting_player and type(a) is c.Shot and action_when_on_court(a, over_player, actions):
                shot_kind = a.kind
                if not dic.has_key(shot_kind):
                    dic[shot_kind] = [0, 0]
                    
                dic[shot_kind][0] += 1
                if a.made:
                    dic[a.kind][1] += 1
    return dic      


def fg_stats(shots_dic):
    ''' '''
    attempted, made = 0, 0
    for i in shots_dic:
        attempted += shots_dic[i][0]
        made += shots_dic[i][1]
        
    return attempted, made
    
    
def specific_shot_stats(shot_type='Jump Shot', players=['Felton'], team='NYK', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' '''
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    res = []
    for actions in actions_per_game:
        this_game = [0, 0]
        for a in actions:
            for p in players:
                if a.player == p and type(a) is c.Shot:         
                    if shot_type == a.kind:
                        this_game[0] += 1   
                        if a.made:
                            this_game[1] += 1
        res.append(this_game)
    return res

        
def shots_made_raster(players=['Duncan'], team='SAS', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' plots a made shots raster. Y axis is the number of the game. X axis is the minute in the game''' 
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    shots, played_games = [], []
    i = 0
    for actions in actions_per_game:
        i += 1
        for a in actions:
            for p in players:
                if a.player == p and type(a) is c.Shot:
                    if a.made:
                        shots.append(a.time)
                        played_games.append(i)     

    return shots, played_games


def actions_raster(players=['Duncan'], team='SAS', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results', action=c.Shot):
    ''' creates an actions raster. Y axis is the number of the game. X axis is the minute in the game''' 
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    final_actions, played_games = [], []
    i = 0
    for actions in actions_per_game:
        i += 1
        for a in actions:
            for p in players:
                if a.player == p and type(a) is action:
                    final_actions.append(a.time)
                    played_games.append(i)     
    return final_actions, played_games


def actions_histogram(actions, bin_size=30.0):
    ''' take actions and return histogram based on minutes.'''
    actions_np = []
    for a in actions:
        actions_np.append(int(float(a)))
    h = np.zeros(np.max(np.array(actions_np))/bin_size+1)
    for a in actions:
        h[int(float(a)/bin_size)] += 1
    return h

def time_raster(player='Duncan', team='SAS', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' plots a made time raster. Y axis is the number of the game. X axis is the minute in the game'''
    actions_per_game = tf.relevant_actions(team, path, s_date=s_date, f_date=f_date)
    time_list, played_games = [], []
    i = 0
    for actions in actions_per_game:
        i += 1
        toc = time_on_court(player, actions)
        #print g, toc
        if len(toc) != 0:
            for t in toc:
                time_list += tf.break_it(t)
            played_games.append(i)
    return time_list, played_games
    

def shots4time(player='Noah', team='CHI', s_date='20091001', f_date='20151231', path='D:\Gal\Work\Results'):
    ''' '''
    actions_1, played_games_1 = actions_raster(players=[player], team=team, s_date=s_date, f_date=f_date, path=path, action=c.Shot)
    actions_2, played_games_2 = time_raster(player, team, s_date, f_date, path)
    actions_3, played_games_3 = actions_raster(players=[player], team=team, s_date=s_date, f_date=f_date, path=path, action=c.FreeThrow)

    mpl.hold(1)
    #mpl.title(player+','+team)
    #mpl.plot(actions_1, played_games_1, 'b.')
    #mpl.plot(actions_3, played_games_3, 'm.')

    mpl.figure(1)
    mpl.plot(actions_histogram(actions_1), 'b')
    mpl.plot(actions_histogram(actions_2), 'r')
    mpl.plot(actions_histogram(actions_3), 'm')

    mpl.figure(2)
    mpl.title(player+','+team)
    mpl.figtext(0, 0.5, 'blue - shots \nred - presentce on court\nmagenta - freethrows')
    mpl.plot(tf.normalize(tf.movingAverage(actions_histogram(actions_1), 3)), 'b')
    mpl.plot(tf.normalize(tf.movingAverage(actions_histogram(actions_2), 3)), 'r')
    mpl.plot(tf.normalize(tf.movingAverage(actions_histogram(actions_3), 3)), 'm')
    mpl.xlabel('Time (in half minutes)')
    mpl.ylabel('Points')
    mpl.grid(axis='x')
    mpl.xticks(np.arange(1, 5)*Q_LEN/30)
    mpl.show()


def action_timing(actions):
    t_last, q = 0, 0
    result = []

    for a in actions:
        t = float(a.time)

        if t > quarter_time(q):
            t_last = quarter_time(q)
            q += 1
            result.append(c.ActionTiming(a, t-t_last))

        elif len(result) != 0:

            if type(a) in [c.Block, c.Steal, c.Assist]:
                result.append(c.ActionTiming(a, result[-1].get_timing()))

            else:
                result.append(c.ActionTiming(a, t-t_last))

        if type(a) in [c.Rebound, c.Steal, c.Turnover, c.Shot, c.FreeThrow, c.Foul]:
            if type(a) is c.Shot:
                if a.made:
                    t_last = t

            elif type(a) is c.Foul and result[-1].get_timing() > 10:
                    t_last = 14 + t
            else:
                t_last = t

        if 24 > result[-1].get_timing() + quarter_time(q) - t:
            new_t = 24 - quarter_time(q) + t
            result[-1].set_timing(new_t)

    return result


def action_raster24(player='Duncan', team='SAS', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results', action=c.Shot):
    games = tf.specific_team(team, path)
    actions, played_games = [], []
    i = 0
    for g in games:
        h = g.find('-')
        if g[:h] > s_date and g[:h] < f_date:
            i += 1
            im = tf.read_from_file(path+'\\'+g)
            data = tf.create_data(im[0], im[1], im[2])
            at = action_timing(data)
            for t in at:
                if type(t.get_action()) is action and t.get_action().player == player:
                    actions.append(t)
                    played_games.append(i)
    return actions, played_games


def get_timing_histogram(action_timing_array):
    result = []
    for t in action_timing_array:
        result.append(t.get_timing())
    return result


def time_out_histogram(team='NYK', path='D:\Gal\Work\Results'):
    ''' '''
    timeouts = []
    games = tf.specific_team(team, path)
    for g in games:
        h = g.find('-')
        im = tf.read_from_file(path+'\\'+g)
        if g[h+1:h+4] == team:
            timeouts += tf.find_timeouts(im[0], im[1])
        else:
            timeouts += tf.find_timeouts(im[0], im[2])
    mpl.plot(actions_histogram(timeouts))
    mpl.show()

    return timeouts


def timeout_affection(timeouts, times, home, away, effect_time=120):
    ''' IN PROGRESS '''
    before, after = [], []
    for t in timeouts:
        start, stop = -1, -1
        i = 0
        while i < len(times)and start < 0:
            if np.abs(t-times[i]) < effect_time:
                start = i
            i += 1

        while i < len(times)and stop < 0:
            if np.abs(t-times[i]) > effect_time:
                stop = i
            i += 1

        t_index = int(np.where(times == t)[0][0])
        before.append(scores(away[start:t_index], home[start:t_index]))
        after.append(scores(away[t_index:stop], home[t_index:stop]))
    return before, after


def good_shots_checking(players=['James'], team='MIA', s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' '''
    d = shooting_stats_over_games(players, team, s_date, f_date, path)
    rank = 0
    total_shots = 0
    total_baskets = 0
    for k in d.keys():
        rank += float(d[k][1]/d[k][0]) * d[k][1]
        total_shots += d[k][0]
        total_baskets += d[k][1]
    return float(rank/total_shots), float(total_baskets/total_shots)


def good_shots_checking_given_shots(shots, players=['James']):
    d = shooting_stats(shots, players)
    rank = 0
    total_shots = 0
    total_baskets = 0
    for k in d.keys():
        rank += float(d[k][1]/d[k][0]) * d[k][1]
        total_shots += d[k][0]
        total_baskets += d[k][1]
    return float(rank/total_shots), float(total_baskets/total_shots)


def reb_after_ft(times, home, away):
    ''' This function finds how mays offensive reb comes after missing a free_throws'''
    h_def_reb, h_off_reb = 0, 0
    a_def_reb, a_off_reb = 0, 0
    for i in xrange(len(times)-1):
        if "Free Throw" in home[i] and "Missed" in home[i]:
            if "Rebound" in home[i+1] and not "Team" in home[i+1]:
                h_off_reb += 1
            elif "Rebound" in away[i+1]:
                a_def_reb += 1
        elif "Free Throw" in away[i] and "Missed" in away[i]:
            if "Rebound" in home[i+1]:
                h_def_reb += 1
            elif "Rebound" in away[i+1] and not "Team" in away[i+1]:
                a_off_reb += 1
    return h_def_reb, h_off_reb, a_def_reb, a_off_reb


def reb_after_ft_per_team(team='SAS', path='D:\Gal\Work\Results'):
    '''IN PROGRESS '''
    home_games = tf.specific_team(team, home=True, away=False)
    away_games = tf.specific_team(team, home=False, away=True)
    off_reb, allow_off_reb = 0, 0
    for g in home_games:
        times, home, away = tf.read_from_file(path+r'\\'+g)
        h_def_reb, h_off_reb, a_def_reb, a_off_reb = reb_after_ft(times, home, away)
        allow_off_reb += a_off_reb
        off_reb += h_off_reb
    for g in away_games:
        times, home, away = tf.read_from_file(path+r'\\'+g)
        h_def_reb, h_off_reb, a_def_reb, a_off_reb = reb_after_ft(times, home, away)
        allow_off_reb += h_off_reb
        off_reb += a_off_reb

    return off_reb, allow_off_reb, len(home_games)+len(away_games)


def starting_5_shots(team='SAS', starting_5=['Parker', 'Green', 'Leonard', 'Duncan', 'Splitter'], s_date='10000101', f_date='21001231', path='D:\Gal\Work\Results'):
    ''' '''
    colors = ['r', 'b', 'g', 'm', 'k']
    actions, played_games = [0]*5, [0]*5
    mpl.hold(1)
    for i in xrange(0, 5):
        actions[i], played_games[i] = action_raster24(starting_5[i], team, s_date, f_date, path)
        mpl.plot(actions_histogram(get_timing_histogram(actions[i]), 1), colors[i])
    mpl.xlabel('Time (in seconds)')
    mpl.grid(axis='x')
    mpl.xticks([0, 12, 24])
    mpl.xlim([0, 26])
    mpl.show()


#________________________________
# MAIN 
#________________________________

#p = tf.get_text('http://www.nba.com/games/20140514/BKNMIA/gameinfo.html')


#im = tf.import_data(p)
#s = shooting_stats_over_games(['Duncan'],'SAS',f_date= '20110701')
#for i in s :
#   print i, s[i], float(s[i][1])/s[i][0]*100
#DEBUG
#for i in xrange(len(im[0])) :
#   print i, im[0][i], im[1][i], im[2][i]
#DEBUG
#im = tf.read_from_file(r'D:\Gal\Work\Results\20101005-DETMIA.txt')
#s = scores(im[2], im[1])
#g = tf.create_game_from_file('20101005-DETMIA.txt')
#plot_scores(g)
#plot_diff(g)

'''
t_1 = time_out_histogram('MIA')
t_2 = time_out_histogram('SAS')
t_3 = time_out_histogram('NYK')

mpl.hold(1)
mpl.figure(1)
mpl.plot(actions_histogram(t_1))
mpl.plot(actions_histogram(t_2))
mpl.plot(actions_histogram(t_3))
mpl.show()
'''
#tf.write_data(im[0], im[1], im[2], im[3])
#print starting5(p)
#data = tf.create_data(im[0], im[1], im[2])
#print time_on_court('James', data)
#shots= []
#for d in data:
#    if type(d) is c.Shot :
#        shots.append(d)
#mpl.plot(time_raster2())
#plot_scores(im[0], im[1], im[2], im[3])
