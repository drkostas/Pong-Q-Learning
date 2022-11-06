"""
usage: python3 runEpisode.py q f
"""
import random
import sys
import time
import numpy as np
from pong.pongclass import pongGame


def load_args():
    """ Loads the arguments from the command line"""
    args = sys.argv[1:]
    if len(args) == 2:
        # Load the arguments
        grid_dem, file_name = args[:2]
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    try:
        grid_dem = int(grid_dem)
        file_name = str(file_name)
    except ValueError:
        raise Exception("Invalid Parameter Types")

    include_vel = True

    return grid_dem, file_name


def tab(item, grid_dem):
    val = int(np.floor((item/300)*grid_dem))
    if (val == grid_dem):
        val = val-1
    return val


def tab_vel(vel):
    if (vel > 0):
        return 1
    else:
        return 0


def main():
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    grid_dem, file_name = load_args()
    load_path = "Q_tables/"+file_name+".npy"
    include_vel = True

    # Load Games and Q table
    p = pongGame(300, 300, draw=True, game_speed=2)
    Q = np.load(load_path)

    # Start the game
    print("------ Games Starts ------")
    done = False
    hits = 0
    while (not done):
        state = p.getState()[:6]
        player, c, ball_x, ball_y, vel_x, vel_y = state
        player = tab(player, grid_dem)
        ball_x = tab(ball_x, grid_dem)
        ball_y = tab(ball_y, grid_dem)
        if include_vel:
            vel_x = tab_vel(vel_x)
            vel_y = tab_vel(vel_y)

        if include_vel:
            action_values = Q[player, ball_x, ball_y, vel_x, vel_y]
        else:
            action_values = Q[player, ball_x, ball_y]

        action = np.argmax(action_values)
        r = p.takeAction(action)
        if (r == 1):
            hits = hits+1
        p.draw()
        time.sleep(0.05)
        if (r == 100 or r == -100):
            done = True
    if (r == 100):
        print("The agent WON with "+str(hits)+" hits.")
    else:
        print("The agent LOST with " + str(hits) + " hits.")


if __name__ == "__main__":
    main()
