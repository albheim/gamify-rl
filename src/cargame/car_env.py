import gym
import pygame
from gym import error, spaces, utils
from gym.utils import seeding

class CarEnv(gym.Env):
    """
    Description of environment, rewards, actions...
    """
    metadata = {'render.modes': ['human']} # human, others???

    def __init__(self):
        pygame.init()
        self.width = 400
        self.height = 300
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.x = 0
        self.y = 0

    def step(self, action):
        """
        Args: action
        Returns: state, reward, done, info
        """
        if action == 0:
            self.y += 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.x -= 1
        elif action == 4:
            pass

        return (self.x, self.y), 0, False, {}

    def reset(self):
        """
        Returns: state
        """
        self.x = self.width / 2
        self.y = self.height / 2

        return (self.x, self.y)

    def render(self, mode='human', close=False):
        """
        Args: mode, close
        Returns: ???
        """
        pygame.display.flip()
        self.screen.fill((0,0,0))
        pygame.draw.rect(self.screen, (0, 128, 255), pygame.Rect(self.x, self.y, 30, 20))
        

    def close(self):
        """
        Close window if viewer is open
        """
        








