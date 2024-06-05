from sklearn.manifold import MDS
from agent import Agent
from solution import checkSubtour, Combination, optimizer, Combination2
import csv
import time

# V = 17
# L = 4
# # d = [
# # [9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 7, 3, 5, 8, 8, 5],
# #  [3, 9999, 3, 48, 48, 8, 8, 5, 5, 7, 7, 3, 19, 3, 8, 8, 5],
# #  [5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
# #  [48, 48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
# #  [48, 48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
# #  [8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 12, 12, 8],
# #  [8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 12, 12, 8],
# #  [5, 5, 26, 12, 12, 8, 8, 9999, 12, 5, 5, 5, 5, 26, 8, 8, 7],
# #  [5, 5, 26, 12, 12, 8, 8, 12, 9999, 5, 5, 5, 5, 26, 8, 8, 7],
# #  [3, 7, 3, 48, 48, 8, 8, 5, 5, 9999, 14, 3, 0, 3, 8, 8, 5],
# #  [3, 7, 3, 48, 48, 8, 8, 5, 5, 14, 9999, 3, 0, 3, 8, 8, 5],
# #  [7, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5],
# #  [3, 19, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5],
# #  [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24],
# #  [8, 8, 50, 6, 6, 12, 12, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8],
# #  [8, 8, 50, 6, 6, 12, 12, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8],
# #  [5, 5, 26, 12, 12, 8, 8, 7, 7, 5, 5, 5, 5, 26, 8, 8, 9999]
# # ]

# d = [
# [99999, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
#       [548, 99999, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
#       [776, 684, 99999, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
#       [696, 308, 992, 99999, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
#       [582, 194, 878, 114, 99999, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
#       [274, 502, 502, 650, 536, 99999, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
#       [502, 730, 274, 878, 764, 228, 99999, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
#       [194, 354, 810, 502, 388, 308, 536, 99999, 342, 388, 730, 468, 354, 320, 662, 742, 856],
#       [308, 696, 468, 844, 730, 194, 194, 342, 99999, 274, 388, 810, 696, 662, 320, 1084, 514],
#       [194, 742, 742, 890, 776, 240, 468, 388, 274, 99999, 342, 536, 422, 388, 274, 810, 468],
#       [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 99999, 878, 764, 730, 388, 1152, 354],
#       [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 99999, 114, 308, 650, 274, 844],
#       [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 99999, 194, 536, 388, 730],
#       [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 99999, 342, 422, 536],
#       [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 99999, 764, 194],
#       [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 99999, 798],
#       [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 99999],
# ]

# mapper = MDS(n_components=2)
# map = mapper.fit_transform(d)

def only1SEC(V, L, d, q, Q, map, gamma, decay, file):
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
    S1 = Combination(orphans2, V, L, opt2[1], opt2[2])

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
            # check if new subtours are created after applying the SEC
            orphans = checkSubtour(V, L, solution[1], solution[2])
            newActions = Combination(orphans, V, L, solution[1], solution[2])
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
            state = nextState
            totalIterations += 1
            if len(agent.memory) >= batch:
                agent.tuning(batch)
        if len(agent.memory) != 0:
            agent.memory[len(agent.memory) - 1][4] = True
        if agent.epsilon <= agent.epsilon_min:
            end = time.time()
            duration = end - start
            file = open("1only.csv", mode='a', newline='')
            writer = csv.writer(file)
            writer.writerow([gamma, decay, totalIterations, duration])
            break
            