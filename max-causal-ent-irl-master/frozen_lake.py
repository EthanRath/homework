"""
Module from the assignments for UC Berkeley's Deep RL course.
"""

import numpy as np
import sys
from six import StringIO, b
import copy

import torch
from torch.nn.functional import softmax as sm
from gym import utils
import discrete_env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "5x5": [
        "FFFFF",
        "FFFFF",
        "SFFFF",
        "FFFFF",
        "FFFFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

class FrozenLakeEnv(discrete_env.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5x5",is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile

class BinGame(discrete_env.DiscreteEnv):
    def __init__(self):
        nS = 2
        nA = 2
        self.states = [0,1]
        self.actions = [0,1]
        isd = [.5, .5]
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            for a in range(nA):
                P[s][a].append([1, (s+a)%2, s == 0, False])
        self.desc = ""
        super(BinGame, self).__init__(nS, nA, P, isd)
                
        

class EthanCommute(discrete_env.DiscreteEnv):
    """

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = ""
        locations = ["home", "work"]
        morning_weather = ["clear", "raining", "cloudy", "windy"]
        to_weather = ["clear", "raining", "cloudy", "windy"]
        from_Weather = ["clear", "raining", "cloudy", "windy"]
        umb_states = ["umb", "no_umb"]
        bike_states = ["bike", "no_bike"]
        base_prob = {"clear":1, "raining": .05, "windy": .05, "cloudy": .1}
        
        actions = ["T+umb", "T-umb", "Bike+umb", "Bike-umb"]
        
        nS = (len(locations)*len(morning_weather)*len(to_weather)*len(from_Weather)*len(umb_states)*len(bike_states))+1
        nA = len(actions)
        
        states = [[]]
        text_states = [[]]
        total_prob = [1]
        
        spaces = [locations, morning_weather, to_weather, from_Weather, umb_states, bike_states]
        for j in range(len(spaces)):
            space = spaces[j]
            possibilities = []
            p2 = []
            probs = np.ones(len(space))
            for i in range(len(space)):
                if space[i] == "work": probs[i] = 0
                if space[i] == "no_umb" or space[i] == "no_bike": probs[i] = 0
                if j == 1 and space[i] in morning_weather: probs[i] = base_prob[space[i]]
                
                temp = [0 for i in range(len(space))]
                temp[i] = 1
                possibilities.append(temp)
                p2.append(space[i])

            new_states = []
            nts = []
            new_prob = []

            for i in range(len(states)):
                for j in range(len(possibilities)):
                    new_states.append(states[i] + possibilities[j])
                    nts.append(text_states[i] +[p2[j]])
                    cur_state = nts[-1]
                    if len(cur_state) >=2 and (cur_state[-1] in morning_weather and cur_state[-2] in morning_weather):
                        if cur_state[-2] == "cloudy":
                            new_prob.append(total_prob[i]*probs[j] * (.3 if cur_state[-1] in ["raining", "windy"] else .2))     
                        else:
                            new_prob.append(total_prob[i]*probs[j] * (.9 if cur_state[-1] == cur_state[-2] else .1))   
                            
                    else:
                        new_prob.append(total_prob[i]*probs[j])   
                    
            states = new_states
            text_states = nts
            total_prob = new_prob
            
        fixed_states = []
        fixed_text = []
        fixed_prob = []
        for i in range(len(states)):
            if "home" in text_states[i] and ("no_umb" in text_states[i] or "no_bike" in text_states[i]):
                continue
            fixed_states.append(states[i])
            fixed_text.append(text_states[i])
            fixed_prob.append(total_prob[i])
        
        states = fixed_states
        text_states = fixed_text
        total_prob = fixed_prob
        
        #add final, done state
        states.append([0 for i in range(len(states[0]))])
        text_states.append(["done"])
        total_prob.append(0)
        
        isd = np.array(total_prob)
        #print(isd)
        isd /= isd.sum()
        #print(isd.sum())
        
        # for i in range(len(states)):
        #     if not ("no_umb" in text_states[i] or "no_bike" in text_states[i]):
        #         print(f"{states[i]} {text_states[i]} {isd[i]}")
        
        
        nS = len(states)
        nA = len(actions)
                
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        # P[s,a] = [[prob, new_state, rewards, done]]
        
        def get_cur_weather(index):
            offset = 2 if text_states[index] == "home" else 3
            return text_states[index][offset]
        
        self.rmat = np.zeros((nS, nA))
        #print(self.rmat.shape)
        for i in range(nS):
            for j in range(nA):
                if i == nS-1:
                    P[i][j].append([1, i, 0, True])
                    continue
                next_state = copy.copy(states[i])
                next_state[0] = 1 - next_state[0]; next_state[1] = 1 - next_state[1] #switch location
                rew = 1 #base reward of 1 for biking in clear/cloudy weather
                
                weather = get_cur_weather(i)
                if "Bike" in actions[j]:
                    if weather == "windy": rew = -0.5
                    if weather == "rainy": rew = -1.5
                    if "no_bike" in text_states[i]: rew -= 1.5 #penalty for taking blue-bike
                else: 
                    next_state[-1] = 1; next_state[-2] = 0 #no bike available at next step
                    if weather == "rainy":
                        if "+umb" in actions[j] and "umb" in text_states[i]: rew = 1 #happy to avoid the rain
                        else: rew = -1 #walking in the rain is just as bad
                    else: rew = 0 #I'd rather bike if it's not raining
                if "-umb" in actions[j]:
                    next_state[-3] = 1; next_state[-4] = 0 #no umbrella available at next step
                    #rew += .1 #small bonus for having to carry less
                
                if "work" in text_states[i]:
                    P[i][j].append([1, len(states)-1, rew, True])
                else:
                    P[i][j].append([1, states.index(next_state), rew, False])
                self.rmat[i,j] = rew
        #print(P)
        super(EthanCommute, self).__init__(nS, nA, P, isd)
        self.states = states
        self.text_states = text_states
        
def don_care_ethan(env):
    st = env.text_states
    #actions = ["T+umb", "T-umb", "Bike+umb", "Bike-umb"]
    policy = np.zeros((len(st), 4))    
    policy[:, 3] = 1
    return policy

def ethan_belief(env):
    st = env.text_states
    belief = np.zeros((len(st), len(st)))
    
    for i in range(len(st)):
        for j in range(len(st)):
            if st[i][0] == "done":
                belief[i][i] += 1 if i==j else 0
            if st[j][0] == "done":
                continue#belief[i,j] = 0 if st[i][0] != st[j][0] else 0
            elif "home" in st[i]:
                if st[i][1] == "clear":
                    belief[i,j] += 1 if st[j][2] in ["clear", "cloudy"] else 0
                    belief[i,j] += 1 if st[j][3] in ["clear", "cloudy"] else 0
                elif st[i][1] in ["raining", "cloudy"]:
                    belief[i,j] += 1 if st[j][2] in ["raining", "windy"] else 0
                    belief[i,j] += 1 if st[j][3] in ["raining", "windy"] else 0
                elif st[i][1] == "cloudy":
                    belief[i,j] += 1 
                if st[i][1] != st[j][1] or st[i][0] != st[j][0]:
                    belief[i,j] = 0 
            elif "work" in st[i]:
                if st[i][2] == "clear":
                    belief[i,j] += 1 if st[j][3] in ["clear", "cloudy"] else 0
                elif st[i][2] in ["raining", "cloudy"]:
                    belief[i,j] += 1 if st[j][3] in ["raining", "windy"] else 0
                elif st[i][2] == "cloudy":
                    belief[i,j] += 1 
                if st[i][2] != st[j][2] or st[i][0] != st[j][0] or st[i][1] != st[j][1]:
                    belief[i,j] = 0 
    belief /= .05
    belief = torch.tensor(belief)
    belief = sm(belief, dim = 1)
    return belief
                    
def cautious_ethan(env):
    st = env.text_states
    #actions = ["T+umb", "T-umb", "Bike+umb", "Bike-umb"]
    policy = np.zeros((len(st), 4))
    for i in range(len(st)):
        if st[i][0] == "done":
            policy[i] = policy[i] + 1 / 4
        elif "home" in st[i]:
            if st[i][1] in ["raining", "windy"]:
                policy[i][0] = 1; policy[i][1] = 0
            elif st[i][1] == "cloudy":
                policy[i][0] = .75; policy[i][2] = .25
            else:
                policy[i][2] = 1; policy[i][3] = 0
        else:
            if "no_bike" in st[i]:
                policy[i][0] = 1; policy[i][1] = 0
            elif st[i][2] in ["raining", "windy"]:
                policy[i][0] = 1; policy[i][1] = 0
            elif st[i][2] == "cloudy":
                policy[i][0] = .75; policy[i][2] = .25
            else:
                policy[i][2] = 1; policy[i][3] = 0
    return policy

if __name__ == "__main__":
    
    mdp = BinGame()
    print(mdp.P)
    #print(mdp.P)
    
    #print(ethan_belief(mdp))
