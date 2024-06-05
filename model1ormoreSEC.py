from sklearn.manifold import MDS
from agent import Agent
from solution import checkSubtour, Combination, optimizer, Combination2
import itertools
import csv
import time

def powerset(nodeList):
    Q = []
    for i in range(2, len(nodeList)):
        Q.append(list(itertools.combinations(nodeList, i)))
    result = []
    for el in nodeList:
        result.append(el)
    for elem in Q:
        for tup in elem:
            result.append(list(tup))
    # print(result)
    return result

def moreThan1SEC(V, L, d, q, Q, map, gamma, decay, file):
    totaldist = 0
    for i in range(V):
        for j in range(V):
            if d[i][j] != 9999:
                totaldist += d[i][j]

    S1 = []

    opt = optimizer([], V, L, map, d, q, Q)
    orphans1 = checkSubtour(V, L, opt[1], opt[2])
    S_init = Combination(orphans1, V, L, opt[1], opt[2])

    opt2 = optimizer(S_init, V, L, map, d, q, Q)
    orphans2 = checkSubtour(V, L, opt2[1], opt2[2])
    S1 = powerset(Combination(orphans2, V, L, opt2[1], opt2[2]))

    agent = Agent(S1, len(S1), gamma, decay)
    batch = 10
    episodes = 10000
    totalIterations = 0
    start = time.time()

    for episode in range(episodes):
        SEC = S_init[:]
        state = [-1, len(orphans2)]
        # print(len(orphans2))
        reward = 0
        for iter in range(agent.seclen):
            # perform action
            action = agent.action(state[0])
            if type(S1[action][0]) == list:
                for elem in S1[action]:
                    if elem not in SEC:
                        SEC.append(elem)
            else:
                SEC.append(S1[action])
            solution = optimizer(SEC, V, L, map, d, q, Q)
            print(SEC)
            # check if new subtours are created after applying the SEC
            orphans = checkSubtour(V, L, solution[1], solution[2])
            newActions = powerset(Combination(orphans, V, L, solution[1], solution[2]))
            # with new created subtours, new actions can be performed
            for elem in newActions:
                if elem not in S1:
                    agent.action_size += 1
                    S1.append(elem)
            agent.S = S1
            agent.model = agent.transferLearning()
            nextState = [action, len(orphans)]
            # get reward for action
            if nextState[1] == 0:
                reward += (totaldist - solution[0]) / 1000
                agent.remember(state[0], action, reward, nextState[0], True, solution[0], -1, S_init)
                if (iter + 1) < agent.seclen:
                    agent.seclen = iter + 1
                break
            else:
                agent.remember(state[0], action, reward, nextState[0], False, solution[0], -1, S_init)
            totalIterations += 1
            state = nextState
            if len(agent.memory) >= batch:
                agent.tuning(batch)
        if len(agent.memory) != 0:
            agent.memory[len(agent.memory) - 1][4] = True
        if agent.epsilon <= agent.epsilon_min:
            end = time.time()
            duration = end - start
            file = open("1orMore.csv", mode='a', newline='')
            writer = csv.writer(file)
            writer.writerow([gamma, decay, totalIterations, duration])
            break
        
def moreThan1SECtest(V, L, d, q, Q, map, gamma, decay, file, obj, name):
    totaldist = 0
    for i in range(V):
        for j in range(V):
            if d[i][j] != 9999:
                totaldist += d[i][j]

    S1 = []

    opt = optimizer([], V, L, map, d, q, Q)
    orphans1 = checkSubtour(V, L, opt[1], opt[2])
    S_init = Combination(orphans1, V, L, opt[1], opt[2])

    opt2 = optimizer(S_init, V, L, map, d, q, Q)
    orphans2 = checkSubtour(V, L, opt2[1], opt2[2])
    S1 = powerset(Combination(orphans2, V, L, opt2[1], opt2[2]))

    agent = Agent(S1, len(S1), gamma, decay)
    batch = 10
    episodes = 10000
    modelObj = 0

    for episode in range(episodes):
        SEC = S_init[:]
        state = [-1, len(orphans2)]
        # print(len(orphans2))
        reward = 0
        for iter in range(agent.seclen):
            # perform action
            action = agent.action(state[0])
            if type(S1[action][0]) == list:
                for elem in S1[action]:
                    if elem not in SEC:
                        SEC.append(elem)
            else:
                SEC.append(S1[action])
            solution = optimizer(SEC, V, L, map, d, q, Q)
            # print(SEC)
            # check if new subtours are created after applying the SEC
            orphans = checkSubtour(V, L, solution[1], solution[2])
            newActions = powerset(Combination(orphans, V, L, solution[1], solution[2]))
            # with new created subtours, new actions can be performed
            for elem in newActions:
                if elem not in S1:
                    agent.action_size += 1
                    S1.append(elem)
            agent.S = S1
            agent.model = agent.transferLearning()
            nextState = [action, len(orphans)]
            # get reward for action
            if nextState[1] == 0:
                modelObj = solution[0]
                reward += (totaldist - solution[0]) / 1000
                agent.remember(state[0], action, reward, nextState[0], True, solution[0], -1, S_init)
                if (iter + 1) < agent.seclen:
                    agent.seclen = iter + 1
                break
            else:
                agent.remember(state[0], action, reward, nextState[0], False, solution[0], -1, S_init)
            state = nextState
            if len(agent.memory) >= batch:
                agent.tuning(batch)
        if len(agent.memory) != 0:
            agent.memory[len(agent.memory) - 1][4] = True
        if agent.epsilon <= agent.epsilon_min:
            file = open("1orMore.csv", mode='a', newline='')
            writer = csv.writer(file)
            totaldem = 0
            for elem in q:
                totaldem += elem
            writer.writerow([name, V, L, totaldem, Q, obj, modelObj])
            break
            