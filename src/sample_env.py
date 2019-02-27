import gym
from gym import error, space, utils
from gym.utils import seeding

class SampleEnv(gym.Env):
    """
    Description of environment, rewards, actions...
    """
    metadata = {'render.modes': ['human']} # human, others???

    def __init__(self):
        pass

    def step(self, action):
        """
        Args: action
        Returns: state, reward, done, info
        """
        pass

    def reset(self):
        """
        Returns: state
        """
        pass

    def render(self, mode='human', close=False):
        """
        Args: mode, close
        Returns: ???
        """
        pass

    def close(self):
        """
        Close window if viewer is open
        """
        pass
