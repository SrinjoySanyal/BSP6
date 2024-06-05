import numpy
import keras
import math
import random

class Agent:
    def __init__(self, S, actionSize, gamma, decay):
        self.S = S
        # self.state_size = stateSize
        self.action_size = actionSize
        self.memory = []
        self.gamma = gamma #0.65
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = decay #0.75
        self.model = self.model()
        self.explore = 0
        self.seclen = 1000
        
    def model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(50, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(50, input_dim=50, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='relu'))
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        return model
    
    def transferLearning(self):
        newModel = keras.Sequential()
        for layers in self.model.layers[:-1]:
            newModel.add(layers)
        newModel.add(keras.layers.Dense(self.action_size, activation='relu'))
        newModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        return newModel
    
    def action(self, state):
        if numpy.random.rand() <= self.epsilon:
            print("yes")
            next = numpy.random.randint(self.action_size)
            while next == state:
                print("same state")
                next = numpy.random.randint(self.action_size)
            iter = 0
            while [state, next, 0, next, True] in self.memory and iter <= 100:
                print("same random")
                next = numpy.random.randint(self.action_size)
                iter += 1
            return next
        print("no")
        q_values = self.model.predict(numpy.array([state]))
        return numpy.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done, modelObj, startState, initSet):
        self.memory.append([state, action, reward, next_state, done])
        if reward > 0 and done == True:
            print("optim found")
            print("Deep RL solution: ", modelObj)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            for j in range(10):
                self.memory.append([state, action, reward, next_state, done])
        f = open("memory.txt", "w")
        content = ""
        for elem in self.memory:
            if elem[0] != startState:
                arr = [self.S[elem[0]], elem[1], elem[2], self.S[elem[3]], elem[4]]
            else:
                arr = [initSet, elem[1], elem[2], self.S[elem[3]], elem[4]]
            content += (str(arr) + "\n")
        f.write(content)
        # print(self.memory)
        f.close()
        if len(self.memory) % math.factorial(self.action_size) >= self.explore and self.explore != 0:
            self.explore = math.factorial(self.action_size)
            self.seclen += 1
        
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
