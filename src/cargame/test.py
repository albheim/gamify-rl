from car_env import CarEnv
import pygame

env = CarEnv()

state = env.reset()
close_screen = False

while True:
    action = 4

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN: action = 0
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT: action = 1
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP: action = 2
        if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT: action = 3
        if event.type == pygame.QUIT:
            close_screen = True
    next_state, reward, done, info = env.step(action)
    env.render()

    if done or close_screen:
        break
pygame.display.quit()
pygame.quit()

