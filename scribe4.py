import numpy as np
from matplotlib import pyplot as plt


class mdp:
    def __init__(self, q):
        self.q = q
        self.state = np.random.randint(0, 2)
    def __call__(self, action):
        self.state = self.state + action % 2
        #simulate channel noise
        if np.random.rand() > self.q:
            return 1-self.state
        return self.state
    
class agent:
    def __init__(self,q):
        self.q = q
        self.belief = [.5, .5]
        self.prev_action = None
    def update(self, y):
        if self.prev_action == 0:
            self.belief[0] *= self.q if y == 0 else 1-self.q
            self.belief[1] *= self.q if y == 1 else 1-self.q
        else:
            self.belief[0] = self.belief[1] * self.q if y == 0 else self.belief[1] * (1-self.q)
            self.belief[1] = self.belief[0] * self.q if y == 1 else self.belief[0] * (1-self.q)
    def act(self):
        return 0 if self.belief[0] <= .5 else 1
    

def run_sims(num_samples, depth, q = .8):
    data = {}
    for i in range(num_samples):
        m = mdp(q)
        a = agent(q)
        action = 0
        z = ""
        for j in range(depth):
            y = m(action)
            a.update(y)
            action = a.act()

            if z not in data.keys():
                data[z] = {0: 0, 1:0}
            data[z][y] += 1
            z += str(y)
    return data

def calc_entropy(data):
    keys = list(data.keys())
    keys.sort(key = lambda x: len(x))
    cur_length = 0
    values = [0]

    table = {}
    total = 0 #data[[]][0] + data[[]][1]
    for key in keys:
        if len(key) > cur_length:
            for key2 in table.keys():
                p_z_0 = table[key2][0]/total
                p_z_1 = table[key2][1]/total
                p_z = (table[key2][0] + table[key2][1]) / total
                values[cur_length] -= p_z_0*np.log2(p_z_0/p_z)
                values[cur_length] -= p_z_1*np.log2(p_z_1/p_z)
            cur_length += 1
            table = {}
            total = 0
            values.append(0)

        total += data[key][0] + data[key][1]
        table[key] = data[key]

    for key2 in table.keys():
        p_z_0 = table[key2][0]/total
        p_z_1 = table[key2][1]/total
        p_z = (table[key2][0] + table[key2][1]) / total
        values[cur_length] -= p_z_0*np.log2(p_z_0/p_z)
        values[cur_length] -= p_z_1*np.log2(p_z_1/p_z)

    return values

if __name__ == "__main__":

    data = run_sims(500_000, 12)
    values = calc_entropy(data)
    print(values)

    plt.figure(dpi = 150)
    plt.plot(values)
    plt.xlabel("timesteps t")
    plt.ylabel("Conditional Entropy H(Y_t | Z_t)")
    plt.title("Conditonal Entropy Over Time")
    plt.savefig("polot.png")
    plt.ylim(0, 1)
    plt.close()
    
