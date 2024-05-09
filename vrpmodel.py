import gurobipy
from gurobipy import GRB
import itertools
from sklearn.manifold import MDS
import matplotlib.pyplot
import matplotlib.colors
import keras
import numpy
import random
import math

def Combination(n, set):
    Q = []
    
    for i in range(2, n):
        Q.append(list(itertools.combinations(set, i)))
    
    result = []
    for i in Q:
        for j in i:
            result.append(list(j))
    
    return result

def optimizer(S, V, L, map, d):
    try:
        matplotlib.pyplot.scatter(map[1:,0], map[1:,1])
        matplotlib.pyplot.plot(map[0, 0], map[0, 1], "ro")
        m = gurobipy.Model("vrp", env=env)
        #create variables
        x = {}
        for l in range(L):
            for i in range(V):
                for j in range(V):
                    if i == j:
                        x[i, j, l] = m.addVar(lb=0, ub=0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    elif d[i][j] == 0:
                        # if there is no edge from i to j then d[i][j] = 0
                        x[i, j, l] = m.addVar(lb=0, ub=0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    elif i != j:
                        x[i, j, l] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))

        #set objective function
        m.setObjective(gurobipy.quicksum([d[i][j]*x[i,j,l] for i in range(V) for j in range(V) for l in range(L)]), GRB.MINIMIZE)

        #set constraints
        for l in range(L):
            m.addConstr(gurobipy.quicksum([x[i,0,l] for i in range(1, V)]) == 1, "c%s"%l)
            m.addConstr(gurobipy.quicksum([x[0,j,l] for j in range(1, V)]) == 1, "c2t%s"%l)
            
        for i in range(1, V):
            m.addConstr(gurobipy.quicksum([x[i,j,l] for j in range(V) for l in range(L)]) == 1, "c3t%sin%s"%(l,i))
                
        for j in range(1, V):
            m.addConstr(gurobipy.quicksum([x[i,j,l] for i in range(V) for l in range(L)]) == 1, "c4t%sout%s"%(l,j))
            
        for i in range(1, V):
            for l in range(L):
                m.addConstr(gurobipy.quicksum([x[i,j,l] for j in range(V)]) == gurobipy.quicksum([x[j, i, l] for j in range(V)]))

        for subtour in S:
            m.addConstr(gurobipy.quicksum(x[i,j,lorry] for i in subtour for j in subtour if i != j for lorry in range(L)) <= len(subtour) - 1, "sec%s"%subtour)
                
        m.write("VRP-DFJ.lp")

        m.optimize()
        status = m.Status
        print("Solution status: ", status)
        if status == GRB.OPTIMAL:
            all_vars = m.getVars()
            values = m.getAttr("X", all_vars)
            names = m.getAttr("VarName", all_vars)

            for name, val in zip(names, values):
                for t in range(L):
                    for i in range(V):
                        for j in range(V):
                            if val != 0 and "(%s,%s,%s)"%(i, j, t) in name:
                                matplotlib.pyplot.plot([map[i][0], map[j][0]], [map[i][1], map[j][1]], color=colors[t])
                                matplotlib.pyplot.axis("off")
                            
        matplotlib.pyplot.savefig("map.png")
        matplotlib.pyplot.close()
        return [m.getAttr("ObjVal"), names, values]
    
    except AttributeError:
        matplotlib.pyplot.scatter(map[1:,0], map[1:,1])
        matplotlib.pyplot.plot(map[0, 0], map[0, 1], "ro")
        m = gurobipy.Model("vrp", env=env)
        #create variables
        x = {}
        for l in range(L):
            for i in range(V):
                for j in range(V):
                    if i == j:
                        x[i, j, l] = m.addVar(lb=0, ub=0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    elif d[i][j] == 0:
                        # if there is no edge from i to j then d[i][j] = 0
                        x[i, j, l] = m.addVar(lb=0, ub=0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    elif i != j:
                        x[i, j, l] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))

        #set objective function
        m.setObjective(gurobipy.quicksum([d[i][j]*x[i,j,l] for i in range(V) for j in range(V) for l in range(L)]), GRB.MINIMIZE)

        #set constraints
        for l in range(L):
            m.addConstr(gurobipy.quicksum([x[i,0,l] for i in range(1, V)]) == 1, "c%s"%l)
            m.addConstr(gurobipy.quicksum([x[0,j,l] for j in range(1, V)]) == 1, "c2t%s"%l)
            
        for i in range(1, V):
            m.addConstr(gurobipy.quicksum([x[i,j,l] for j in range(V) for l in range(L)]) == 1, "c3t%sin%s"%(l,i))
                
        for j in range(1, V):
            m.addConstr(gurobipy.quicksum([x[i,j,l] for i in range(V) for l in range(L)]) == 1, "c4t%sout%s"%(l,j))
            
        for i in range(1, V):
            for l in range(L):
                m.addConstr(gurobipy.quicksum([x[i,j,l] for j in range(V)]) == gurobipy.quicksum([x[j, i, l] for j in range(V)]))

        for subtour in S:
            m.addConstr(gurobipy.quicksum(x[i,j,lorry] for i in subtour for j in subtour if i != j for lorry in range(L)) <= len(subtour) - 1, "sec%s"%subtour)
                
        m.write("VRP-DFJ.lp")

        m.optimize()
        status = m.Status
        names = ""
        values = ""
        print("Solution status: ", status)
        if status == GRB.OPTIMAL:
            all_vars = m.getVars()
            values = m.getAttr("X", all_vars)
            names = m.getAttr("VarName", all_vars)

            for name, val in zip(names, values):
                for t in range(L):
                    for i in range(V):
                        for j in range(V):
                            if val != 0 and "(%s,%s,%s)"%(i, j, t) in name:
                                matplotlib.pyplot.plot([map[i][0], map[j][0]], [map[i][1], map[j][1]], color=colors[t])    
                                matplotlib.pyplot.axis("off")           
        matplotlib.pyplot.savefig("map.png")
        matplotlib.pyplot.close()
        return [1000000, names, values]

def go(start, looped, V, L, names, values):
    if start == 0:
        return looped
    else:
        for name, val in zip(names, values):
            for t in range(L):
                for j in range(V):
                    if name == "x(%s,%s,%s)"%(start, j, t) and val != 0:
                        looped[j] = 1
                        # print(looped)
                        return go(j, looped, V, L, names, values)

def checkSubtour(V, L, names, values):
    looped = [0 for i in range(V)]
    toreach = []
    for name, val in zip(names, values):
        for t in range(L):
            for j in range(V):
                if name == "x(%s,%s,%s)"%(0,j,t) and val != 0:
                    looped[j] = 1
                    toreach.append(j)
                    
    for elem in toreach:
        looped = go(elem, looped, V, L, names, values)
    return [i for i in range(V) if looped[i] == 0]
    
class Agent:
    def __init__(self, S):
        self.state_size = len(S)
        self.action_size = len(S)
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.75
        self.model = self.model()
        
    def model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(50, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(50, input_dim=50, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        return model
    
    def action(self, state):
        if numpy.random.rand() <= self.epsilon:
            return numpy.random.randint(self.action_size)
        q_values = self.model.predict(numpy.array([state]))
        return numpy.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done, objVal, modelObj):
        self.memory.append([state, action, reward, next_state, done])
        if reward > 0 and done == True:
            print("Deep RL solution: ", objVal)
            print("Optimal solution: ", modelObj)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            for j in range(100):
                goodpath = []
                for i in range(len(self.memory) - 1, 0, -1):
                    goodpath.append(self.memory[i])
                    self.memory[i][2] = reward
                    if self.memory[i][0] == 0:
                        break
                for elem in goodpath[::-1]:
                    self.memory.append(elem)    
        f = open("memory.txt", "w")
        content = ""
        for elem in self.memory:
            content += str(elem) + "\n"
        f.write(content)
        f.close()
        
    def tuning(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for elem in minibatch:
            state = elem[0]
            action = elem[1]
            reward = elem[2]
            next_state = elem[3]
            done = elem[4]
            if not done:
                target = reward + self.gamma * numpy.max(self.model.predict(numpy.array([next_state]))[0])
            else:
                target = reward
            Q = self.model.predict(numpy.array([state]))
            Q[0][action] = target
            self.model.fit(x=numpy.array([state]), y=Q)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

options = {
    "WLSACCESSID": "9aadbf35-dba3-4158-b503-3b68b4e0d6dd",
    "WLSSECRET": "2cd0ea41-6f53-419b-8696-c93718870444",
    "LICENSEID": 2489877,
}

env = gurobipy.Env(params=options)
colors = ["red", "yellow", "green", "blue", "black"]

V = 17
L = 3
d = [
 [9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
 [3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
 [5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
 [48, 48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
 [48, 48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
 [8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
 [8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
 [5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0],
 [5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0],
 [3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5],
 [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5],
 [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5],
 [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5],
 [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24],
 [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8],
 [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8],
 [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999]
]

nodesMin0 = list(range(1, V))
S = Combination(V - 1, nodesMin0)

mapper = MDS(n_components=2)
map = mapper.fit_transform(d)

totaldist = 0
for i in range(V):
    for j in range(V):
        if d[i][j] != 9999:
            totaldist += d[i][j]

objective = optimizer(S, V, L, map, d)

agent = Agent(S)
batch = 30
episodes = 2000

for episode in range(episodes):
    done = False
    itnum = 0
    SEC = []
    state = 0
    reward = 0
    opt = optimizer(SEC, V, L, map, d)
    subtours = checkSubtour(V, L, opt[1], opt[2])
    result = opt[0] + 100*len(subtours)**2
    if len(subtours) == 0:
        reward += opt[0]
    while not done and itnum <= len(S):
        act = agent.action(state)
        # next = S[act] 
        SEC.append(S[act])
        opt = optimizer(SEC, V, L, map, d)
        subtours = checkSubtour(V, L, opt[1], opt[2])
        result1 = opt[0] + 100*len(subtours)**2
        if result1 < result and itnum <= len(S):
            if len(subtours) == 0:
                reward += (totaldist - opt[0])
                done = True
            result = result1
            agent.remember(state, act, reward, act, done, opt[0], objective[0])
            state = act
            itnum += 1
            if len(agent.memory)>batch:
                agent.tuning(batch)
        else:
            if len(agent.memory) > 0:
                agent.memory[len(agent.memory)-1][4] = True
                done = True     
    if agent.epsilon <= agent.epsilon_min:
        print("Deep RL solution: ", opt[0])
        print("Optimal solution: ", objective[0])
        break