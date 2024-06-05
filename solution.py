import gurobipy
from gurobipy import GRB
import itertools
import matplotlib.pyplot
import matplotlib.colors

options = {
    "WLSACCESSID": "9aadbf35-dba3-4158-b503-3b68b4e0d6dd",
    "WLSSECRET": "2cd0ea41-6f53-419b-8696-c93718870444",
    "LICENSEID": 2489877,
}

env = gurobipy.Env(params=options)
colors = ["red", "yellow", "green", "blue", "black"]

def Combination2(n, set):
    Q = []

    for i in range(2, n):
        Q.append(list(itertools.combinations(set, i)))

    result = []
    for i in Q:
        for j in i:
            result.append(list(j))

    return result

def Combination(setNodes, V, L, names, values):
    Q = []
    while setNodes != []:
        looped = [0 for i in range(V)]
        start = 0
        for name, val in zip(names, values):
            for t in range(L):
                for j in range(V):
                    if name == "x(%s,%s,%s)"%(setNodes[0],j,t) and val != 0:
                        looped[j] = 1
                        start = j
        looped = go(start, looped, V, L, names, values, end = setNodes[0])
        tour = [i for i in range(V) if looped[i] == 1]
        if tour != []:
            Q.append(tour)
        setNodes = [i for i in setNodes if i not in tour]
    return Q

def optimizer(S, V, L, map, d, q, Q):
    print("SEC: ", S)
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
                        x[i, j, l] = m.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    # elif d[i][j] == 0:
                    #     # if there is no edge from i to j then d[i][j] = 0
                    #     x[i, j, l] = m.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))
                    elif i != j:
                        x[i, j, l] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x(%s,%s,%s)"%(i,j,l))

        #set objective function
        m.setObjective(gurobipy.quicksum([d[i][j]*x[i,j,l] for i in range(V) for j in range(V) for l in range(L) if i != j]), GRB.MINIMIZE)

        #set constraints
        totaldem = 0
        for elem in q:
            totaldem += elem
        
        if totaldem <= Q*L:  
            for l in range(L):
                m.addConstr(gurobipy.quicksum([x[i,0,l] for i in range(1, V)]) == 1, "c%s"%l)
                m.addConstr(gurobipy.quicksum([x[0,j,l] for j in range(1, V)]) == 1, "c2t%s"%l)
        else:
            print("No depot constraint")
        
        for i in range(1, V):
            m.addConstr(gurobipy.quicksum([x[j, i, l] for j in range(V) for l in range(L)]) == 1, "assign1_to_arc_truck_%s"%l)
            m.addConstr(gurobipy.quicksum([x[i, k, l] for k in range(V) for l in range(L)]) == 1, "assign2_to_arc_truck_%s"%i)    
            
        for j in range(1, V):
            for i in range(1, V):
                if i != j:
                    m.addConstr(gurobipy.quicksum([x[i, j, l] for l in range(L)]) == gurobipy.quicksum([x[j, k, l] for k in range(1, V) for l in range(L) if k != i]))

        for subtour in S:
            m.addConstr(gurobipy.quicksum(x[i,j,lorry] for i in subtour for j in subtour for lorry in range(L) if i != j) <= len(subtour) - 1, "sec%s"%subtour)
            
        for l in range(L):
            m.addConstr(gurobipy.quicksum([x[i,j,l]*q[j] for i in range(V) for j in range(1, V) if i != j]) <= Q, "capconstr%s"%l)
        
                
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
        totaldem = 0
        for elem in q:
            totaldem += elem
        
        if totaldem <= Q*L:
            for l in range(L):
                m.addConstr(gurobipy.quicksum([x[i,0,l] for i in range(1, V)]) == 1, "c%s"%l)
                m.addConstr(gurobipy.quicksum([x[0,j,l] for j in range(1, V)]) == 1, "c2t%s"%l)
            
        for i in range(1, V):
            m.addConstr(gurobipy.quicksum([x[j, i, l] for j in range(V) for l in range(L)]) == 1, "assign1_to_arc_truck_%s"%l)
            m.addConstr(gurobipy.quicksum([x[i, j, l] for j in range(V) for l in range(L)]) == 1, "assign2_to_arc_truck_%s"%i)    
            
        for j in range(1, V):
            for i in range(1, V):
                m.addConstr(gurobipy.quicksum([x[i, j, l] for l in range(L)]) == gurobipy.quicksum([x[j, k, l] for k in range(V) for l in range(L) if k != i]))

        for subtour in S:
            m.addConstr(gurobipy.quicksum(x[i,j,lorry] for i in subtour for j in subtour for lorry in range(L) if i != j) <= len(subtour) - 1, "sec%s"%subtour)
            
        for l in range(L):
            m.addConstr(gurobipy.quicksum([q[j]*x[i,j,l] for i in range(V) for j in range(1, V) if i != j]) <= Q)
                
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

def go(start, looped, V, L, names, values, end):
    # print(start, end)
    if start == end:
        return looped
    else:
        for name, val in zip(names, values):
            for t in range(L):
                for j in range(V):
                    if name == "x(%s,%s,%s)"%(start, j, t) and val != 0:
                        looped[j] = 1
                        # print(looped)
                        return go(j, looped, V, L, names, values, end)

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
        looped = go(elem, looped, V, L, names, values, 0)
    return [i for i in range(V) if looped[i] == 0]