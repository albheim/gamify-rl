import numpy as np
import itertools
import pandas as pd
import plotting
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple
from easy21 import Easy21Env

env = Easy21Env()

def featurize(state, action):
    feat = np.zeros((3, 6, 2))
    for di, d in enumerate([(0, 3), (3, 6), (6, 9)]):
        for pi, p in enumerate([(10, 15), (13, 18), (16, 21), (19, 24), (22, 27), (25, 30)]):
            if state[0] > p[0] and state[0] < p[1] and state[1] > d[0] and state[1] < d[1]:
                feat[di, pi, action] = 1
    return feat.flatten()

def create_epsilon_greedy_policy(Q, nA, eps):
    def policy(state):
        action_probs = np.ones(nA) * eps / nA
        action = 0 if Q(state, 0) > Q(state, 1) else 1
        action_probs[action] += (1 - eps)
        return action_probs
    return policy

def sarsa(env, n_episodes, discount=1.0, alpha=0.01, epsilon=0.05, lambda0=0.8, Qtrue=None):
    nA = env.action_space.n
    obs = env.observation_space.spaces
    nS = (obs[0].n, obs[1].n)

    theta = np.zeros(36)
    def Q(state, action):
        return featurize(state, action).dot(theta)

    policy = create_epsilon_greedy_policy(Q, nA, epsilon)

    episode_error = np.zeros(n_episodes)

    for i in range(n_episodes):
        E = np.zeros(36)

        state = env.reset()

        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)),
                                  p=action_probs)
        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)

            td_target = reward
            if not done:
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)),
                                               p=next_action_probs)

                td_target += discount * Q(next_state, next_action)

            td_error = td_target - Q(state, action)
            E = discount * lambda0 * E + featurize(state, action)
            theta += alpha * td_error * E

            print("\r{} @ {}/{}".format(t, i + 1, n_episodes), end="")

            if done:
                break

            action = next_action
            state = next_state

        episode_error[i] = 0
        Qv = np.zeros(Qtrue.shape)
        for s1 in range(nS[0]):
            for s2 in range(nS[1]):
                s = (s1, s2)
                for a in range(nA):
                    Qv[s][a] = Q(s, a)
                    episode_error[i] += (Q(s, a) - Qtrue[s][a])**2

    print()
    return Qv, episode_error

def mc_create_epsilon_greedy_policy(Q, nA):
    def policy(state, eps):
        action_probs = np.ones(nA) * eps / nA
        action = np.argmax(Q[state])
        action_probs[action] += (1 - eps)
        return action_probs
    return policy

def mc(env, n_episodes, discount=1.0, N0=100):

    nA = env.action_space.n
    obs = env.observation_space.spaces
    nS = (obs[0].n, obs[1].n)

    Q = np.zeros((nS[0], nS[1], nA))
    N = np.zeros((nS[0], nS[1], nA))

    policy = mc_create_epsilon_greedy_policy(Q, nA)

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

Qtrue, _, _ = mc(env, 1000000)
sqerrs = []
lambdas = np.arange(0, 1.01, 0.1)
for lambda0 in lambdas:
    Q, err = sarsa(env, 1000, 1.0, 0.1, 0.05, lambda0, Qtrue)
    sqerrs.append(err)

plt.plot(sqerrs[0])
plt.plot(sqerrs[-1])
plt.title("Q mse over episodes")
plt.xlabel("episode")
plt.ylabel("Q mse")
plt.legend(["lambda=0", "lambda=1"])
plt.show()

plt.plot(lambdas, [err[-1] for err in sqerrs])
plt.title("Q mse for different lambda")
plt.xlabel("lambda")
plt.ylabel("Q mse")
plt.show()

plotting.plot_value_function(np.amax(Qtrue, axis=2))
plotting.plot_value_function(np.amax(Q, axis=2))
