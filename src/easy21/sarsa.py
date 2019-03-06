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

def mc(env, n_episodes, discount=1.0, N0=100):

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

def sarsa(env, n_episodes, discount=1.0, N0=100, lambda0=0.8, Qtrue=None):

    nA = env.action_space.n
    obs = env.observation_space.spaces
    nS = (obs[0].n, obs[1].n)

    Q = np.zeros((nS[0], nS[1], nA))
    N = np.zeros((nS[0], nS[1], nA))

    policy = create_epsilon_greedy_policy(Q, nA)

    episode_error = np.zeros(n_episodes)

    for i in range(n_episodes):
        E = np.zeros((nS[0], nS[1], nA))

        state = env.reset()

        epsilon = N0 / (N0 + sum(N[state]))
        action_probs = policy(state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)),
                                  p=action_probs)
        for t in itertools.count():
            N[state][action] += 1
            next_state, reward, done, _ = env.step(action)

            td_target = reward
            if not done:
                epsilon = N0 / (N0 + sum(N[next_state]))
                next_action_probs = policy(next_state, epsilon)
                next_action = np.random.choice(np.arange(len(next_action_probs)),
                                               p=next_action_probs)

                td_target += discount * Q[next_state][next_action]

            td_error = td_target - Q[state][action]
            E[state][action] += 1

            for s1 in range(nS[0]):
                for s2 in range(nS[1]):
                    for a in range(nA):
                        Q[s1, s2, a] += td_error * E[s1, s2, a] / N[s1, s2, a]
                        E[s1, s2, a] *= discount * lambda0


            print("\r{} @ {}/{} ({})".format(t, i + 1, n_episodes, (reward, epsilon)), end="")

            if done:
                break

            action = next_action
            state = next_state

        episode_error[i] = np.sum((Q-Qtrue)**2)

    print()
    return Q, episode_error

Qtrue, _, _ = mc(env, 10)
Q, err = sarsa(env, 100000, 1.0, 100, 0.7, Qtrue)

# Qtrue, _, _ = mc(env, 1000000)
# sqerrs = []
# lambdas = np.arange(0, 1.01, 0.1)
# for lambda0 in [0.5]:#lambdas:
#     Q, err = sarsa(env, 10000, 1.0, 100, lambda0, Qtrue)
#     sqerrs.append(err)

plt.plot(err)
#plt.plot(sqerrs[0])
#plt.plot(sqerrs[-1])
plt.title("Q mse over episodes")
plt.xlabel("episode")
plt.ylabel("Q mse")
plt.legend(["lambda=0", "lambda=1"])
plt.show()

# plt.plot(lambdas, [err[-1] for err in sqerrs])
# plt.title("Q mse for different lambda")
# plt.xlabel("lambda")
# plt.ylabel("Q mse")
# plt.show()

# plotting.plot_value_function(np.amax(Qtrue, axis=2))
plotting.plot_value_function(np.amax(Q, axis=2))
