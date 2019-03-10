from car_env import CarEnv
import pygame

env = CarEnv()

state = env.reset()

for i in range(100):
    action = 4

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN: action = 0
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT: action = 1
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP: action = 2
        if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT: action = 3

    next_state, reward, done, info = env.step(action)

    if done:
        break

