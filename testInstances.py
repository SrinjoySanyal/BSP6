from sklearn.manifold import MDS
from agent import Agent
from solution import checkSubtour, Combination, optimizer, Combination2
from model1ormoreSEC import moreThan1SECtest
from model1SEC import only1SEC
import csv
import vrplib
import numpy

testInst = [["A/A-n32-k5.vrp", "A/A-n32-k5.sol", 5, 32], 
            ["A/A-n33-k5.vrp", "A/A-n33-k5.sol", 5, 33],
            ["A/A-n33-k6.vrp", "A/A-n33-k5.sol", 6, 33],
            ["A/A-n34-k5.vrp", "A/A-n34-k5.sol", 5, 34],
            ["A/A-n36-k5.vrp", "A/A-n36-k5.sol", 5, 36]
            ]

file = open("test.csv", mode='w', newline='')
writer = csv.writer(file)
writer.writerow(["instance name",
                 "number of nodes", 
                 "number of trucks",
                 "total demands",
                 "capacity for each truck", 
                 "objective value", 
                 "optimal objective value", "Gap"])

gamma = 0.65
decay = 0.45
for elem in testInst:
    instance = vrplib.read_instance(elem[0])
    solution = vrplib.read_solution(elem[1])

    name = instance["name"]
    map = instance["node_coord"]
    d = instance["edge_weight"].tolist()
    # print(d)
    obj = solution["cost"]
    Q = instance["capacity"]
    q = instance["demand"].tolist()
    print(q)
    print(Q)
    L = elem[2]
    V = elem[3]
    
    res = optimizer([], V, L, map, d, q, Q)
    # print(res)
    
    # moreThan1SECtest(V, L, d, q, Q, map, gamma, decay, file, obj, name)