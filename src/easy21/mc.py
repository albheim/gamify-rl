import numpy as np
import itertools
import pandas as pd
import plotting
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple
from easy21 import Easy21Env

env = Easy21Env()

def create_epsilon_greedy_policy(Q, nA):
    def policy(state, eps):
        action_probs = np.ones(nA) * eps / nA
        action = np.argmax(Q[state])
        action_probs[action] += (1 - eps)
        return action_probs
    return policy

def mc(env, n_episodes, discount=1.0, alpha=0.5, N0=100):

    nA = env.action_space.n
    obs = env.observation_space.spaces
    nS = (obs[0].n, obs[1].n)

    Q = np.zeros((nS[0], nS[1], nA))
    N = np.zeros((nS[0], nS[1], nA))

    policy = create_epsilon_greedy_policy(Q, nA)

    episode_reward = np.zeros(n_episodes)
    episode_length = np.zeros(n_episodes)

    for i in range(n_episodes):
        state = env.reset()

        episode = []
        for t in itertools.count():
            epsilon = N0 / (N0 + sum(N[state]))
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)),
                                           p=action_probs)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, reward, action))

            episode_reward[i] += reward
            episode_length[i] = t

            print("\r{} @ {}/{} ({})".format(t, i + 1, n_episodes, episode_reward[i]), end="")

            if done:
                break

            state = next_state

        G = 0
        for state, reward, action in episode[::-1]:
            G = reward + discount * G

        for state, reward, action in episode:
            N[state][action] += 1
            Q[state][action] += (G - Q[state][action]) / N[state][action]
            G = (G - reward) / discount

    print()
    return Q, episode_reward, episode_length

Q, rewards, lengths = mc(env, 800000)

plt.plot(pd.Series(rewards).rolling(10000, min_periods=10000).mean())
plt.show()

plotting.plot_value_function(np.amax(Q, 2))
plotting.plot_value_function(np.argmax(Q, 2), title="Policy function")
