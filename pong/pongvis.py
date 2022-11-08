from pongclass import pongGame
import time


p = pongGame(300, 300, draw=True, game_speed=2)
action = 2
done = False
while (not done):
    r = p.takeAction(action)
    p.draw()
    time.sleep(0.03)
    if (r == 100 or r == -100):
        done = True
    state = p.getState()
    y1, y2, xball, yball, ballHDirection, ballVDirection = state
    if (yball > y1+p.paddle_length/2):
        action = 1
    elif (yball < y1+p.paddle_length/2):
        action = 0
    else:
        action = 2
