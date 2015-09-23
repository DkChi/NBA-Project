# -*- coding: utf-8 -*-
import urllib2
import re
import numpy as np
import matplotlib.pyplot as mpl
from bs4  import BeautifulSoup as bs
import os
import sys
if 'D:\Gal\Work' not in sys.path :
    sys.path.append('D:\Gal\Work')
import classes4NBA as c

# link structure :  http://www.nba.com/games/  DATE: YYYYMMDD  / AwayTeam HomeTeam / gameinfo.html
# can be reached as <a href="/games/20140530/INDMIA/gameinfo.html" 
#title="Link to game info for Indiana Pacers vs. Miami Heat">Complete Stats</a> from  [http://www.nba.com/gameline/20140530/

c.Q_LEN = 12*60
c.OT_LEN = 5*60

atlantic_teams = ['BOS', 'BKN', 'NYK', 'PHI', 'TOR']
central_teams = ['CHI', 'IND', 'DET', 'CLE', 'MIL']
southeast_teams = ['ATL', 'CHA', 'MIA', 'ORL', 'WAS']
southwest_teams = ['DAL', 'SAS', 'HOU', 'MEM', 'NOP']
pacific_teams = ['LAL', 'LAC', 'GSW', 'PHX', 'SAC']
northwest_teams = ['DEN', 'MIN', 'OKC', 'POR', 'UTA']
all_teams = atlantic_teams + central_teams + southeast_teams + southwest_teams + pacific_teams + northwest_teams

# ______________________________________
# Technical Functions
# ______________________________________



def get_text(page):
    try:
        return urllib2.urlopen(page).read()
    finally:
        return urllib2.urlopen(page).read()

# now I need a function to run over pages from the type http://www.nba.com/gameline/20140501/


def import_data(text):
    '''
    This function downloads the play by play data from a specific web page
    '''
    text = text.replace('nbaGIPbPLftScore', "nbaGIPbPLft")
    text = text.replace('nbaGIPbPRgtScore', "nbaGIPbPRgt")
    regex_l = '<td class="nbaGIPbPLft">(.+?)nbsp;</td>'
    pattern_l = re.compile(regex_l)
    regex_m = '<td class="nbaGIPbPMid">(.+?) </td>'
    regex_ms = '<td class="nbaGIPbPMidScore">(.+?) <br>'
    pattern_ms = re.compile(regex_ms)
    pattern_m = re.compile(regex_m)
    regex_r = '<td class="nbaGIPbPRgt">(.+?)nbsp;</td>'
    pattern_r = re.compile(regex_r)
    times1 = c.fix_timing(re.findall(pattern_m, text))
    times2 = c.fix_timing(re.findall(pattern_ms, text))
    times, h = merge2sets(times1, [0]*len(times1), times2, [0]*len(times2))
    left = re.findall(pattern_l, text)
    right = re.findall(pattern_r, text)
    
    soup = bs(text)
    title = ''
    if soup.title != None:
        lim = soup.title.string.rfind(' -')
        title = soup.title.string[:lim]

    return times, left, right, title


def get_details(player):
    ''' '''
    p = player.replace(' ', '_').replace('.', '')
    text = get_text('http://www.nba.com/playerfile/' + p.lower())
    regex_height = '<span class="nbaMeters">/ (.+?)m</span>'
    pattern_height = re.compile(regex_height)
    height = re.findall(pattern_height, text)

    return height


def merge2sets(time_1, data_1, time_2, data_2):
    ''' This function merges the two sets of lists into one set'''
    i1, i2 = 0, 0
    data_result, time_result = [], []
    while i1 < len(time_1) or i2 < len(time_2):
        flag = 2
        if i1 == len(time_1):
            flag = 2    
        elif i2 == len(time_2):
            flag = 1    
        elif time_1[i1] < time_2[i2]:
            flag = 1
            
        if flag == 1:
            time_result.append(time_1[i1])
            data_result.append(data_1[i1])
            i1 += 1
        else:
            time_result.append(time_2[i2])
            data_result.append(data_2[i2])
            i2 += 1
            
    return np.array(time_result), data_result


def merge_actions(al1, al2):
    ''' '''
    res = []
    i1, i2, flag = 0, 0, 1
    while i1 < len(al1) or i2 < len(al2):
        flag = 2
        if i1 == len(al1):
            flag = 2
        elif i2 == len(al2):
            flag = 1    
        elif al1[i1] < al2[i2]:
            flag = 1
                    
        if flag == 1 :
            res.append(al1[i1])
            i1 += 1
        else:
            res.append(al2[i2])
            i2 += 1

    return res     


def write_data(times, home, away, title='???', path='D:\Gal\Work\Results'):
    '''This function writes to file a data of a singal game '''
    f = open(path+'\\'+title+'.txt', 'wb')
    for i in xrange(len(times)):
        f.write(str(times[i]) + ',' + home[i] + ',' + away[i] + '\n')
    f.close()


def write_details(players, details, path='D:\Gal\Work\Results'):
    ''' '''
    f = open(path+'\\'+'players_details.txt', 'wb')
    for i in xrange(len(players)):
        details2write = ''
        for d in details[i]:
            details2write += d+','
        f.write(players[i] + ',' + details2write[:-1] + '\n')
    f.close()


def read_details(path='D:\Gal\Work\Results\players_details.txt'):
    ''' '''
    f = open(path, 'rb')
    lines = f.readlines()
    results = {}
    for line in lines:
        l = line
        second_comma = l.find(',', l.find(',')+1)
        l = l[second_comma+1:]
        d = []
        while l.find(',') > 0:
            d.append(l[:l.find(',')])
            l = l[l.find(',')+1:]
        d.append(l[:-1])
        results[line[:second_comma]] = d

    return results


def read_from_file(path):
    '''This function reads the data from file '''
    f = open(path, 'rb')
    lines = f.readlines()
    times, home, away = [], [], []
    for line in lines:
        l = line.find(',')
        r = line.rfind(',')
        times.append(line[:l])
        away.append(line[l+1:r])        
        home.append(line[r+1:-1])
    f.close()
    return np.array(times), home, away
                                                                                                                                                                                                                          

def starting5(text):
    '''This function checks who were the startings fives for a spesific game '''
    regex ='<td id="nbaGIBoxNme" class="b"><a href="/playerfile/(.+?)</a></td>'
    pattern = re.compile(regex)
    two_fives = re.findall(pattern,text)
    fix_2_5 =[]
    for i in two_fives:
        cut = i.find('>')
        fix_2_5.append(i[cut+1:])
    return fix_2_5[:5], fix_2_5[5:]


def find_links_in_page(text):
    ''' 
    This function returns a list of all the links to the game pages from a spesific page
    The links are in format "/games/20140502/TORBKN/gameinfo.html"
    '''
    r = r'<a href="/games/(.+?)</a>'
    res = []
    pat = re.compile(r)
    links = re.findall(pat, text)
    for l in links:
        h = l.find('html')
        res.append(l[:h+len('html')])
    return res


def bring_all_data(s_date='20091001', f_date='20091117'):
    ''' imports the data from NBA.com'''
    date = s_date
    base_link = r'http://www.nba.com/gameline/'
    links = []
    while date != f_date:
        links += find_links_in_page(get_text(base_link+date))
        day = int(date[-2:])
        month = int(date[4:-2])
        year = int(date[:4])
        day += 1
        if day > 31:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
        date = str(year)
        if month/10 == 0:
            date = date+'0'+str(month)
        else:
            date += str(month)    
        if day/10 == 0:
            date += '0'+str(day)
        else:
            date += str(day)
            
    return links


def write_all_data(s_date='20140101', f_date='20140717'):
    ''' '''
    link = r'http://www.nba.com/games/'
    extras = bring_all_data(s_date, f_date)
    for e in extras:
        print link+e
        p = e.rfind('/')
        im = import_data(get_text(link+e))
        write_data(im[0], im[1], im[2], e[:p].replace('/', '-'))
        print e[:p]
        

def create_data(times, home, away):
    '''
    IN PROGRESS
    need to be add :  Timeouts
    '''
    data = []
    for i in xrange(len(times)):
        flag = True
        if home[i] != '&':
            current = home[i]
        elif away[i] != '&':
            current = away[i]                
        else:
            flag = False
        
        if flag:
            current = current[1:-1]
            first_space = current.find(' ')
            player = current[:first_space]
            current = current[first_space+1:]
            
            if "Turnover" in current:
                colon = current.find(':')
                barket = current.find('(')
                data.append(c.Turnover(current[colon+2:barket-1], times[i], player))
                barket = current.find(')')
                current = current[barket+2:]

                if 'ST' in current:
                    data.append(c.Steal(times[i], c.find_second_action(current)))
   
            elif 'Substitution' in current:
                last_space = current.rfind(' ')
                data.append(c.Sub(times[i], player, current[last_space+1:]))
                 
            elif 'shot' in current or 'Shot' in current:
                colon = current.find(':')
                action = c.Shot(current[:colon], times[i], player)
                if 'Made' in current:
                    action.set_status(True)
                else:
                    action.set_status(False)

                data.append(action)

                if 'AST' in current:
                    data.append(c.Assist(times[i], c.find_second_action(current)))

                elif 'BLK' in current:
                    data.append(c.Block(times[i], c.find_second_action(current)))

            elif 'PF' in current:
                colon = current.find(':')
                barket = current.find('(')
                data.append(c.Foul(current[colon+2:barket-1], times[i], player))

            elif 'Free Throw' in current:
                first_space = current.find(' ')
                action = c.FreeThrow(times[i], player)
                if not 'Missed' in current:
                    action.set_status(True)
                else:
                    action.set_status(False)
                data.append(action)

            elif 'Rebound' in current:
                if home[i] != '&':
                    if home[i-1] == '&':
                        kind = 'Def'
                    else:
                        kind = 'Off'
                else:
                    if home[i-1] == '&':
                        kind = 'Off'
                    else:
                        kind = 'Def'
                first_space = current.find(' ')
                data.append(c.Rebound(kind, times[i], player))

    return data         


def specific_team(team, path='D:\Gal\Work\Results', home=True, away=True):
    ''' This function return a list of all the files that that represent game of a specific team '''
    NAME_LEN = 3
    files = os.listdir(path)
    res = []
    for f in files:
        h = f.find('-')
        nxt = h+NAME_LEN+1
        if (f[h+1:nxt] == team and away) or (f[nxt:nxt+NAME_LEN] == team and home):
            res.append(f)
    return res


def all_games_per_team(team, path='D:\Gal\Work\Results', home=True, away=True):
    res = []
    list_of_home_games = specific_team(team, path, home, False)
    list_of_away_games = specific_team(team, path, False, away)
    for g in list_of_home_games:
        res.append(create_game_from_file(g))
        
    for g in list_of_away_games:
        res.append(create_game_from_file(g))
        
    return res
    



def all_players():
    ''' Finds all the players in the NBA'''
    LINK = r'http://stats.nba.com/players.html'
    text = get_text(LINK)
    regex = r'"http://www.nba.com/playerfile/(.+?)</a>'
    pattern = re.compile(regex)
    players_1 = re.findall(pattern, text)
    res = []
    for i in players_1:
        res.append(i[i.rfind('>')+1:])
    return res


def active_players(players):
    ''' Finds all the Active players in the NBA'''
    ac_players, his_players = [], []
    for p in players:
        #print '***', p # DEBUG
        name_for_net = str.lower(p[p.find(' ')+1:]+'_'+p[:p.find(',')])
        name_for_net.replace('.', '')
        try:
            t = get_text(r'http://www.nba.com/playerfile/'+name_for_net)
            if not 'Sorry, Page Not Found' in t:
                #print p
                ac_players.append(p)
        except urllib2.HTTPError:
            #print 'Error :', p
            his_players.append(p)
    return ac_players, his_players    


def movingAverage(arr, s=3):
    ''' Return a moving average of arr, based on the size s '''
    a = []
    for i in arr:
        a.append(float(i))
    a = np.array(a)
    a = np.append(np.array([a[0]]*(s-1)), a)
    a = np.append(a, np.array([a[-1]]*(s-1)))
    window = np.ones(int(s))/float(s)
    b = np.convolve(a, window, 'same')
    return b[s:-(s-1)]


def normalize(t):
    t /= np.max(np.abs(np.array(t)))
    return t


def glue_arrays(a, i=0):
    ''' glue the edge of the arrays, in order to get the times better '''
    if i >= len(a)-1:  # we reached the end of the list
        if type(a[0]) is list:
            return a

    if i == len(a)-2:
        if a[i][1] == a[i+1][0]:
            if i == 0:
                return [a[:i]+[a[i][0], a[i+1][1]]]
            else:
                return a[:i]+[a[i][0], a[i+1][1]]

    if a[i][1] == a[i+1][0]:
        return a[:i]+glue_arrays([[a[i][0], a[i+1][1]]]+a[i+2:], 0)
    else:
        return glue_arrays(a, i+1)


def break_it(a, i=1):
    '''Breaks the list in the Arithmetic progression with the step of i 
    Example: break_it([1,5],1) >>>> [1,2,3,4,5] '''
    
    return list(range(int(a[0]), int(a[1]+1), i))


def sort_and_remove_duplicates(a):
    a = np.sort(a)
    result = []
    for i in xrange(len(a)-1):
        if a[i] != a[i+1]:
            result.append(a[i])

    result.append(a[-1])
    return result


def find_timeouts(times, plays):
    '''Returns when tere was a timeout '''
    result = []
    for i in xrange(len(plays)):
        if plays[i].find('Timeout') > 0:
            result.append(times[i])
    return result


def distance(a, b):
    result = 0
    for i in xrange(np.min([len(a), len(b)])):
        result += (a[i]-b[i])**2
    return result


def bring_players_from_espn(text):
    ''' built for pages like http://espn.go.com/nba/players/_/position/pg '''
    regex = r'"http://espn.go.com/nba/player/_/id(.+?)</a>'
    pattern = re.compile(regex)
    players_1 = re.findall(pattern, text)
    res = []
    for i in players_1:
        res.append(i[i.rfind('>')+1:])
    return res


def leave_these_actions(actions, a=[c.Shot]):
    ''' This fuctions goes aver the a list of actions and leave only the actions which is given as prameter'''
    actions_1 = []
    for i in xrange(len(actions)):
        if type(actions[i]) in  a:
            actions_1.append(actions[i])
       
    return actions_1


def create_game_from_file(f_name, path='D:\Gal\Work\Results'):
    ''' '''
    barket = f_name.find('-') 
    game = c.Game(f_name[barket+4:barket+7],f_name[barket+1:barket+4],f_name[:barket])
    r = read_from_file(path+'\\'+f_name)
    game.loadActions(r[0],r[1],r[2])
    return game
    
    
def all_games_in_spec_dates(games,s_date='10000101', f_date='21001231'):
    result = []
    for g in games:
        if g.date > s_date and g.date < f_date:
            result.append(g)
    return result
    
    
def relevant_actions(team, path='D:\Gal\Work\Results', home=True, away=True,
                     s_date='10000101', f_date='21001231'):
    data = []
    games = all_games_per_team(team, path, home, away)
    relevant_games = all_games_in_spec_dates(games,
                                             s_date=s_date, f_date=f_date)
    for g in relevant_games:
            if g.get_Home_Team() == team:
                data.append(g.get_Home_Actions())
            else:
                data.append(g.get_Away_Actions())
    return data    
    
#________________________________
# MAIN
#________________________________
'''
a_players, h_players = active_players(sort_and_remove_duplicates(all_players()))
d_players = []
for p in a_players:
    comma = p.find(',')
    pl = p[comma+2:] + ' ' + p[:comma]
    d_players.append(get_details(pl))

write_details(a_players, d_players)
'''