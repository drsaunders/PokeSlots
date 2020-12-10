import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from multi_armed_bandit.bandits import BernoulliBandit
from multi_armed_bandit.solvers import Solver, EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling

def prob_str(probas):
    return " ".join(["%.1f%%" % (p*100) for p in probas])

def plot_beta(a,b,jitter=None):
    x = np.linspace(stats.beta.ppf(0.01, a, b),
                    stats.beta.ppf(0.99, a, b), 100)
    
    y = stats.beta.pdf(x, a, b)
    if jitter is not None:
    	x = x + np.random.normal(len(x)) * jitter

    plt.plot(x, y, '-', lw=5, alpha=0.6, label='beta pdf')
    plt.xlabel("Reward probability")
    plt.yticks([])


def plot_betas(solver, jitter=None):
    plt.figure(figsize=(10,6))
    for a,b in zip(solver._as, solver._bs):
        plot_beta(a, b, jitter)
    plt.legend(np.arange(len(solver._as))+1,loc='center left',title="Machine", fontsize=10)


def plot_credible_intervals(the_as, the_bs, alpha):
    intervals = []
    bottoms = []
    for a,b in zip(the_as, the_bs):
        bottom = stats.beta.ppf(alpha/2, a, b)
        top = stats.beta.ppf(1-(alpha/2), a, b)
        interval = top-bottom
        
        bottoms.append(bottom)
        intervals.append(interval)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(the_as))+1,height=intervals, bottom=bottoms, alpha=0.6)
    a = plt.xticks(np.arange(19)+1)
    plt.ylim([0,1])
    plt.xlabel("Machine")
    plt.ylabel("Reward probability")

def topslots(solver, topn=10):
    probas = solver.estimated_probas
    sort_inds = np.argsort(probas)
    slots = np.arange(solver.bandit.n)+1
    wins = np.array(solver._as) - 1
    tries = np.array(solver._as) + np.array(solver._bs) - 2
    for i in sort_inds[-topn:][::-1]:
        print("Slot %s: %.1f%% (%d out of %d)" % (slots[i], probas[i]*100, wins[i], tries[i]))

def print_history(hist):
    for h in hist:
        print("{} {}".format(h[0],h[1]))

class ThompsonSamplingCGC(ThompsonSampling):
    def __init__(self, init_a=1, init_b=1, num_slots=5):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        dummy_bandit = BernoulliBandit(num_slots)

        super(ThompsonSamplingCGC, self).__init__(dummy_bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
        self.history = []

        
    def pick_slot(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        return i
    
    def recommend_slot(self):
        s = self.pick_slot()
        print("I suggest slot {}".format(s+1))
        return s+1
    
    def update_with_reward(self, slot, reward):
        self._as[slot-1] += reward
        self._bs[slot-1] += (1 - reward)
        self.history.append((slot,reward))
        

    def update_with_list(self, slot_rewards):
        for sr in slot_rewards:
            self.update_with_reward(sr[0],sr[1])
    
