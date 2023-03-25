import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment
class BanditEnv:
    def __init__(self, k):
        self.k = k
        #Its not specified if a_i,b_i, are uniformly distributed
        #self.a = np.random.uniform(size=k)
        #self.b = np.random.uniform(size=k)
        self.a = [round(random.random(), 2) for i in range(k)]
        self.b = [round(random.random(), 2) for i in range(k)]
        self.mu_star = np.max((self.a + self.b) / 2)
        
    def pull_arm(self, arm_idx):
        return np.random.uniform(self.a[arm_idx], self.b[arm_idx])

# Define the epsilon-greedy algorithm
class EpsilonGreedy:
    def __init__(self, k, T):
        self.k = k
        self.epsilon = ((np.log(T) * k)**(1/3)) / (T**(1/3))
        self.e_rew = np.zeros(k)
        self.arm_selec = np.zeros(k)
        
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.e_rew)
        
    def update(self, arm_idx, reward):
        self.arm_selec[arm_idx] += 1
        self.e_rew[arm_idx] += (reward - self.e_rew[arm_idx]) / self.arm_selec[arm_idx] + self.epsilon

# Define the Upper Confidence Bound algorithm
class UCB:
    def __init__(self, k, c=2):
        self.k = k
        self.c = np.sqrt(2 * np.log(T) / np.sum(self.arm_selec))
        self.e_rew = np.zeros(k)
        self.arm_selec = np.zeros(k)
        self.t = 0
        
    def select_arm(self):
            t = np.sum(self.N) + 1
            ucb_vals = self.e_rew + self.c * np.sqrt(np.log(self.t) / (self.arm_selec+ 1e-6))
            return np.argmax(ucb_vals)
        
    def update(self, arm_idx, reward):
        self.arm_selec[arm_idx] += 1
        self.e_rew[arm_idx] += (reward - self.e_rew[arm_idx]) / self.arm_selec[arm_idx]