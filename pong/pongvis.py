from pongclass import pongGame
import time


p = pongGame(300, 300, False, 2)
done = False
while (not done):
    r = p.takeAction(2)
    p.draw()
    time.sleep(0.03)
    if (r == 100 or r == -100):
        done = True
