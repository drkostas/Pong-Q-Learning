"""
usage: python runEpisode.py grid_dem file_name
description: Runs a single episode of the game using the Q-table
output: Prints the results of the game
"""

import sys
import time
import pickle
import numpy as np
from pong.pongclass import pongGame

GAME_SPEED = 6


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

    return grid_dem, file_name


def load_agent(file_name):
    # Load pickle file
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    data['grid_dem'] = 8 if data['grid_dem'] is None else data['grid_dem']
    return data['Q'], data['map_size'], data['grid_dem'], data['include_vel']


def clear_line():
    """ Clears the last line in the terminal"""
    print("\033[A                             \033[A")


def tab(item, grid_dem, map_size):
    val = int(np.floor((item/map_size)*grid_dem))
    if val == grid_dem:
        val = val-1
    return val


def tab_vel(vel):
    if vel > 0:
        return 1
    else:
        return 0


def main():
    """ Main function"""
    print(f"{' Initializing ':-^30}")
    # --- Args Loading and Error Checking --- #
    grid_dem, file_name = load_args()
    load_path = "output/"+file_name+".pkl"
    Q, map_size, grid_dem, include_vel = load_agent(load_path)

    # Load Games and Q table
    p = pongGame(map_size, map_size,
                 draw=True, game_speed=GAME_SPEED)

    # Start the game
    clear_line()
    print(f"{' Game Starts ':-^30}")
    print(f"Map Size: {map_size}x{map_size}, Grid: {grid_dem}")
    print(f"Use Velocity: {include_vel}, Init Game Speed: {GAME_SPEED}")
    done = False
    hits = 0
    while not done:
        state = p.getState()[:6]
        player, c, ball_x, ball_y, vel_x, vel_y = state
        player = tab(player, grid_dem, map_size)
        ball_x = tab(ball_x, grid_dem, map_size)
        ball_y = tab(ball_y, grid_dem, map_size)
        if include_vel:
            vel_x = tab_vel(vel_x)
            vel_y = tab_vel(vel_y)

        if isinstance(Q, np.ndarray):
            if include_vel:
                action_values = Q[player, ball_x, ball_y, vel_x, vel_y]
            else:
                action_values = Q[player, ball_x, ball_y]
        else:
            x = np.array([player, c, ball_x, ball_y, vel_x, vel_y])
            action_values = Q.predict(x=x.reshape(1, 6),
                                      verbose=0)[0][0]

        action = np.argmax(action_values)
        r = p.takeAction(action)
        if r == 1:
            hits = hits+1
        p.draw()
        time.sleep(0.05)
        if r == 100 or r == -100:
            done = True
    if r == 100:
        print(f"The agent WON with {hits} hits.")
    else:
        print(f"The agent LOST with {hits} hits.")


if __name__ == "__main__":
    main()
