import numpy as np
from collections import namedtuple

Action = namedtuple('Action', ['time','player'])
Shot = namedtuple('Shot', Action._fields+('kind','made',))
FreeThrow = namedtuple('FreeThrow', Action._fields+('made',))
Rebound = namedtuple('Rebound', Action._fields+('kind',))
Foul = namedtuple('Foul', Action._fields+('kind',))
Turnover = namedtuple('Turnover', Action._fields+('kind',))
Assist = namedtuple('Assist', Action._fields)
Block = namedtuple('Block', Action._fields)
Steal = namedtuple('Steal', Action._fields)
Sub = namedtuple('Sub', Action._fields+('player_in',))

#-------------------------------------------------
class ActionTiming(object):
    ''' This class represent action and its timing (should be used for 24 sec clock) '''

    def __init__(self, action=None, timing=24):
        self.action = action or Action()
        self.timing = timing

    def get_action(self):
        return self.action

    def get_timing(self):
        return self.timing

    def set_action(self, new_action):
        self.action = new_action

    def set_timing(self, new_timing):
        self.timing = new_timing

    def __str__(self):
        return str(self.timing)+':'+str(self.action)

#-------------------------------------------------


Q_LEN = 12*60
OT_LEN = 5*60

def find_second_action(current):
    '''This function finds the second action in a play by play line (actions like assists and blocks and steals) '''
    s = 1
    colon = current.rfind(':')
    if current[colon+1] != ' ':
        s = 0
    current = current[colon+s+1:]
    first_space = current.find(' ')
    return current[:first_space]                                                                                                                                                                                                                            


def fix_timing(times):
    '''This function fixes the time format'''
    q = 1
    new_times = np.abs(np.array(min2time(times))-Q_LEN)
    for i in xrange(len(times)-1):
        if new_times[i] > new_times[i+1]:
            if q < 4:
                new_times[i+1:] += Q_LEN
                q += 1
            else:
                new_times[i+1:] += OT_LEN
    return new_times            


def min2time(t):
    '''Converts the time from string in the format MM:SS to seconds'''
    if isinstance(t, list):
        return [min2time(i) for i in t]
        
    elif type(t) is str:
        p = t.find(':')    
        return 60*int(t[:p])+float(t[p+1:])    
     
     
class Game(object):
    ''' This Class Represent a Single game'''
    
    def __init__(self, homeTeam='HHH',awayTeam='AAA',date='10010101'):
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.date = date
        self.home_actions = []
        self.away_actions = []
    
    def get_Home_Team(self):
        return self.homeTeam
        
    def get_Away_Team(self):
        return self.awayTeam
        
    def get_date(self):
        return self.date
        
    def get_Away_Actions(self):
        return self.away_actions

    def get_Home_Actions(self):
        return self.home_actions         
    
    @staticmethod
    def iter_time_vals(times, home, away):
        # assumption: len is identical
        for i, cur_time in enumerate(times):
            yield cur_time, home[i], away[i]
            
    def loadActions(self, times, home, away):
        ''' I copied this part from the TF file'''
        h_team = True
        for i, cur_time in enumerate(times):
        #for cur_time, cur_home, cur_away in self.iter_time_vals(times, home, away):
            if home[i] != '&':
                current = str.strip(home[i])[:-1]
                h_team = True
                actions = self.home_actions
            elif away[i] != '&':
                current = str.strip(away[i])[:-1]
                h_team = False
                actions = self.away_actions 
            else:
                continue # empty line
            
            first_space = current.find(' ')
            player = current[:first_space]
            current = current[first_space+1:]
            
            
            if 'Turnover' in current:
                colon = current.find(':')
                barket = current.find('(')
                action = Turnover(times[i], player, current[colon+2:barket-1])
                actions.append(action)
                barket = current.find(')')
                current = current[barket+2:]


                if 'ST' in current:
                    a = Steal(times[i], find_second_action(current))
                    if not h_team:
                        self.home_actions.append(a)
                    else: 
                        self.away_actions.append(a)
   
            elif 'Substitution' in current:
                last_space = current.rfind(' ')
                action = Sub(cur_time, player, current[last_space+1:])
                actions.append(action)
                 
            elif 'shot' in current or 'Shot' in current:
                colon = current.find(':')
                made = 'Made' in current
                action = Shot(cur_time, player, current[:colon], made)   
                actions.append(action)    

                if 'AST' in current:
                    actions.append(Assist(cur_time, find_second_action(current)))
                    
                elif 'BLK' in current:
                    a = Block(cur_time, find_second_action(current))
                    if not h_team:
                        self.home_actions.append(a)
                    else: 
                        self.away_actions.append(a)

            elif 'PF' in current:
                colon = current.find(':')
                barket = current.find('(')
                actions.append(Foul(cur_time, player, current[colon+2:barket-1]))

            elif 'Free Throw' in current:
                #first_space = current.find(' ')
                made = 'Missed' not in current
                actions.append(FreeThrow(cur_time, player, made))
                    
                
            elif 'Rebound' in current:
                if home[i] != '&':
                    h_team = True
                    if home[i-1] == '&':
                        kind = 'Def'
                    else:
                        kind = 'Off'
                else:
                    h_team = False
                    if home[i-1] == '&':
                        kind = 'Off'
                    else:
                        kind = 'Def'
                #first_space = current.find(' ')
                actions.append(Rebound(cur_time, player, kind))
                    
            


    def __str__(self):
        return self.date+';\n'+self.homeTeam+':'+self.home_actions+';\n'+self.awayTeam+':'+self.away_actions
        
    def title(self):
        return self.awayTeam+' Vs. '+self.homeTeam+' ; '+self.date
            