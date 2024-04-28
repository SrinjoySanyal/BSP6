import gurobipy
from gurobipy import GRB
import itertools
from sklearn.manifold import MDS
import matplotlib.pyplot
import matplotlib.colors

options = {
    "WLSACCESSID": "9aadbf35-dba3-4158-b503-3b68b4e0d6dd",
    "WLSSECRET": "2cd0ea41-6f53-419b-8696-c93718870444",
    "LICENSEID": 2489877,
}

env = gurobipy.Env(params=options)

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

mapper = MDS(n_components=2)
map = mapper.fit_transform(d)
print(map[1][0])
matplotlib.pyplot.scatter(map[1:,0], map[1:,1])
matplotlib.pyplot.plot(map[0, 0], map[0, 1], "ro")
colors = ["red", "yellow", "green", "blue", "black"]

#subtour elimination
def Combination(n, set):
    Q = []
    
    for i in range(2, n):
        Q.append(list(itertools.combinations(set, i)))
    
    result = []
    for i in Q:
        for j in i:
            result.append(list(j))
    
    return result

Vmin0 = list(range(1, V))
S = Combination(V - 1, Vmin0)

def optimizer(S):
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
                        
    matplotlib.pyplot.savefig("map.png")
    return m.getAttr("ObjVal")