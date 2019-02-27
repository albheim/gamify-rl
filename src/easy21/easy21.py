import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


def draw_card(np_random, black=False):
    if not black:
        black = np.random.randint(3) != 0
    return np.random.randint(10) + 1, black

def sum_hand(hand):
    return sum([value if black else -value for value, black in hand])

def is_bust(hand):
    v = sum_hand(hand)
    return v < 1 or v > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

class Easy21Env(gym.Env):
    """
    Modified version of Blackjack, no aces of facecards
    Infinite deck with 10 values and two colors, uniform over values and 1/3 red 2/3 black
    Start of game both dealer and player draw one black card that is shown
    Player may stick (0) or hit (1), total value is black values minus red values.
    Value over 21 or below 1 means bust with reward -1.
    After sticking the dealer draw until any sum over or equal to 17.
    If player has higher score the reward is +1, if draw it's 0 and if less it's -1
    The state is represented by a tuple of your current value and the dealers
    face up card.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(41), # all values from -9 to 31, implement other space???
            spaces.Discrete(10)))
        self.seed()

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Args: action
        Returns: state, reward, done, info
        """
        assert self.action_space.contains(action)
        if action: # hit
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else: # stick
            done = True
            player_sum = sum_hand(self.player)
            dealer_sum = sum_hand(self.dealer)
            while dealer_sum < 17 and dealer_sum > 0:
                self.dealer.append(draw_card(self.np_random))
                dealer_sum = sum_hand(self.dealer)
            if dealer_sum > 21 or player_sum > dealer_sum:
                reward = 1
            elif player_sum == dealer_sum:
                reward = 0
            else:
                reward = -1
        return self._get_state(), reward, done, {}

    def _get_state(self):
        return (sum_hand(self.player) + 9, sum_hand(self.dealer) - 1)

    def reset(self):
        """
        Returns: state
        """
        self.dealer = [draw_card(self.np_random, True)]
        self.player = [draw_card(self.np_random, True)]
        return self._get_state()

    def render(self, mode='human', close=False):
        """
        Args: mode, close
        Returns: ???
        """
        pass
