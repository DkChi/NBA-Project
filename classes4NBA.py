import numpy as np

class Action(object):
    ''' this class represents an action in the game '''
    
    def __init__(self, time=0, player=''):
        self.time = time
        self.player = player
    
    def get_time(self):
        ''' returns the time of the action'''
        return self.time
        
    def get_player(self):
        return self.player
        
    def set_player(self, new_player):
        self.player = new_player
    
    def set_time(self, new_time):
        self.time = new_time

    def __str__(self):
        return '{}:{}'.format(self.player, self.time)


class Shot(Action):
    ''' This Class represent a shot '''
    def __init__(self, kind, time=0, player='', made=False):
        self.kind = kind
        self.made = made
        super(Shot, self).__init__(time, player)
        #Action.__init__(self, time, player)
    
    def get_kind(self):
        return self.kind
        
    def is_made(self):
        return self.made

    def set_kind(self, new_kind):
        self.kind = new_kind
        
    def set_status(self, status):
        self.made = status

    def __str__(self):
        if self.made:
            m = 'Made'
        else:
            m = 'Missed'
        return str(self.time)+"-"+self.player+' '+self.kind+":"+m


class Assist(Action):
    def __init__(self, time=0, player=''):
        super(Assist, self).__init__(time, player)
        #Action.__init__(self, time, player)


class Foul(Action):
    ''' '''
    def __init__(self, kind, time=0, player=''):
        self.kind = kind
        super(Foul, self).__init__(time, player)
        #Action.__init__(self, time, player)
            
    def get_kind(self):
        return self.kind
        
    def set_kind(self, new_kind):
        self.kind = new_kind


class Rebound(Action):
    ''' '''    
    def __init__(self, kind, time=0, player=''):
        self.kind = kind
        super(Rebound, self).__init__(time, player)
        #Action.__init__(self, time, player)
            
    def get_kind(self):
        return self.kind
        
    def set_kind(self, new_kind):
        self.kind = new_kind


class Sub(Action):        
    ''' '''
    def __init__(self, time=0, player_out='', player_in=''):
        self.player_in = player_in
        Action.__init__(self, time, player_out)
    
    def get_player_in(self):
        return self.player_in
        
    def get_player_out(self):
        return self.player
        
    def set_player_in(self, new_player):
        self.player_in = new_player    
    
    def __str__(self):
        return str(self.time)+" - "+self.player+" replaced by "+self.player_in


class Turnover(Action):
    ''' '''
    def __init__(self, kind, time=0, player=''):
        self.kind = kind
        Action.__init__(self, time, player)
            
    def get_kind(self):
        return self.kind
        
    def set_kind(self, new_kind):
        self.kind = new_kind


class Steal(Action):
    ''' '''           
    def __init__(self, time=0, player=''):
        super(Steal, self).__init__(time, player)
        #Action.__init__(self, time, player)


class Block(Action):
    ''' '''           
    def __init__(self, time=0, player=''):
        super(Block, self).__init__(time, player)
        #Action.__init__(self, time, player)


class FreeThrow(Action):
    ''' '''
    def __init__(self, time=0, player='', made=False):
        self.made = made
        Action.__init__(self, time, player)
        
    def is_made(self):
        return self.made
        
    def set_status(self, status):
        self.made = status
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
#------------------------------------------------

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
    #if type(t) is list:
        return [min2time(i) for i in t]
        
    elif type(t) is str:
        p = t.find(':')    
        return 60*int(t[:p])+float(t[p+1:])    
     
     
class Game(object):
    ''' This Class Represent a Single game'''
    
    def __init__(self, homeTeam='HHH', awayTeam='AAA', date='10010101'):
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
        hTeam = True
        #times = fix_timing(times)
        for i, cur_time in enumerate(times):
        #for i in xrange(len(times)):
        #for cur_time, cur_home, cur_away in self.iter_time_vals(times, home, away):
            if home[i] != '&':
                current = str.strip(home[i])[:-1]
                hTeam = True
                #actions = self.home_actions
            elif away[i] != '&':
                current = str.strip(away[i])[:-1]
                hTeam = False
            else:
                continue # empty line
            
            first_space = current.find(' ')
            player = current[:first_space]
            current = current[first_space+1:]
            
            if "Turnover" in current:
                colon = current.find(':')
                barket = current.find('(')
                a = Turnover(current[colon+2:barket-1], times[i], player)
                if hTeam:
                    self.home_actions.append(a)
                else:
                    self.away_actions.append(a)
                barket = current.find(')')
                current = current[barket+2:]

                if 'ST' in current:
                    a = Steal(times[i], find_second_action(current))
                    if not hTeam:
                        self.home_actions.append(a)
                    else: 
                        self.away_actions.append(a)
   
            elif 'Substitution' in current:
                last_space = current.rfind(' ')
                a = Sub(cur_time, player, current[last_space+1:])
                if hTeam :
                    self.home_actions.append(a)
                else:
                    self.away_actions.append(a)
                 
            elif 'shot' in current or 'Shot' in current:
                colon = current.find(':')
                action = Shot(current[:colon], cur_time, player)
                if 'Made' in current:
                    action.made = True
                else:
                    action.made = False
                    
                if hTeam :
                    self.home_actions.append(action)
                else:
                    self.away_actions.append(action)

                if 'AST' in current:
                    a = Assist(cur_time, find_second_action(current))
                    if hTeam :
                        self.home_actions.append(a)
                    else:
                        self.away_actions.append(a)

                elif 'BLK' in current:
                    a = Block(cur_time, find_second_action(current))
                    if not hTeam:
                        self.home_actions.append(a)
                    else: 
                        self.away_actions.append(a)

            elif 'PF' in current:
                colon = current.find(':')
                barket = current.find('(')
                a = Foul(current[colon+2:barket-1], cur_time, player)
                if hTeam:
                    self.home_actions.append(a)
                else:
                    self.away_actions.append(a)

            elif 'Free Throw' in current:
                #first_space = current.find(' ')
                action = FreeThrow(cur_time, player)
                if not 'Missed' in current:
                    action.made = True
                else:
                    action.made = False
                    
                if hTeam:
                    self.home_actions.append(action)
                else:
                    self.away_actions.append(action)

            elif 'Rebound' in current:
                if home[i] != '&':
                    hTeam = True
                    if home[i-1] == '&':
                        kind = 'Def'
                    else:
                        kind = 'Off'
                else:
                    hTeam = False
                    if home[i-1] == '&':
                        kind = 'Off'
                    else:
                        kind = 'Def'
                #first_space = current.find(' ')
                a = Rebound(kind, cur_time, player)
                if hTeam:
                    self.home_actions.append(a)
                else:
                    self.away_actions.append(a)


    def to_string(self):
        return self.date+';\n'+self.homeTeam+':'+self.home_actions+';\n'+self.awayTeam+':'+self.away_actions
        
    def title(self):
        return self.awayTeam+' Vs. '+self.homeTeam+' ; '+self.date
            