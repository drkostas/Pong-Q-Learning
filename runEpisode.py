"""
usage: python3 runEpisode.py q f
"""
import random
import sys
import time
import numpy as np
from pong.pongclass import pongGame

def main(args):
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    if len(args) == 2:
        # Load the arguments
        q, f = args[:2]
        # TODO:Cast the arguments to the correct type
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    try:
        grid_dem = int(q)
        file_name = str(f)
    except ValueError:
        raise Exception("Invalid Parameter Types")


    include_Vel = True
    p = pongGame(300, 300)
    done = False

    Q = np.load(file_name)

    hits = 0
    x = p.getState()
    while (not done):

        player,c , ball_x, ball_y,vel_x,vel_y = p.getState()[:6]
        player = Tab(player,grid_dem)
        ball_x = Tab(ball_x,grid_dem)
        ball_y = Tab(ball_y,grid_dem)
        if(include_Vel):
            vel_x = Tab_Vel(vel_x)
            vel_y = Tab_Vel(vel_y)


        if (include_Vel):
            action_values = Q[player, ball_x, ball_y, vel_x, vel_y]
        else:
            action_values = Q[player, ball_x, ball_y]


        action = np.argmax(action_values)
        r = p.takeAction(action)
        if(r ==1):
            hits = hits+1
        p.draw()
        time.sleep(0.05)
        if (r == 100 or r == -100):
            done = True
    if(r ==100):
        print("The agent WON with "+str(hits)+" hits.")
    else:
        print("The agent LOST with " + str(hits) + " hits.")

def Tab(item, grid_dem):
    val = int(np.floor((item/300)*grid_dem))
    if(val==grid_dem):
        val = val-1
    return val

def Tab_Vel(vel):
    if(vel>0):
        return 1
    else:
        return 0

if __name__ == "__main__":
    main(sys.argv[1:])
