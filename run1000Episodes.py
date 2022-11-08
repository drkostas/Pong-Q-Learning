"""
usage: python runEpisode.py grid_dem file_name
description: Runs a single episode of the game using the Q-table
output: Prints the results of the game
"""

import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from pong.pongclass import pongGame

GAME_SPEED = 2
DRAW = False


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


def tab(item, grid_dem, map_size) -> int:
        val = int(np.floor((item/map_size)*grid_dem))
        if val >= grid_dem:
            val = grid_dem-1
        return val

def tab_vel_x(vel: float) -> int:
    # return float(f"{vel:.1f}")
    # return int(vel)
    if vel < -2:
        return 0
    elif vel < 0:
        return 1
    elif vel < 2:
        return 2
    else:
        return 3

def tab_vel_y(vel: float) -> int:
    # return float(f"{vel:.1f}")
    # return int(vel)
    if vel < -1.5:
        return 0
    elif vel < -1.0:
        return 1
    elif vel < -0.5:
        return 2
    elif vel < 0:
        return 3
    elif vel < 0.5:
        return 4
    elif vel < 1.0:
        return 5
    elif vel < 1.5:
        return 6
    else:
        return 7
    

def main():
    """ Main function"""
    print(f"{' Initializing ':-^30}")
    # --- Args Loading and Error Checking --- #
    grid_dem, file_name = load_args()
    load_path = "output/"+file_name+".pkl"
    Q, map_size, grid_dem, include_vel = load_agent(load_path)
    # Start the game
    clear_line()
    print(f"{' Game Starts ':-^30}")
    print(f"Map Size: {map_size}x{map_size}, Grid: {grid_dem}")
    print(f"Use Velocity: {include_vel}, Init Game Speed: {GAME_SPEED}")
    rewards = []
    total_hits = []
    total_wins = []
    player_vals = []
    ball_x_vals = []
    ball_y_vals = []
    vel_x_vals = []
    vel_y_vals = []
    for _ in tqdm(range(1000)):
        # Load Game
        p = pongGame(map_size, map_size,
                     draw=DRAW, game_speed=GAME_SPEED)
        done = False
        hits = 0
        while not done:
            state = p.getState()[:6]
            player, c, ball_x, ball_y, vel_x, vel_y = state
            player_vals.append(player)
            ball_x_vals.append(ball_x)
            ball_y_vals.append(ball_y)
            vel_x_vals.append(vel_x)
            vel_y_vals.append(vel_y)
            player = tab(player, grid_dem, map_size)
            ball_x = tab(ball_x, grid_dem, map_size)
            ball_y = tab(ball_y, grid_dem, map_size)
            if include_vel:
                vel_x = tab_vel_x(vel_x)
                vel_y = tab_vel_y(vel_y)

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
            if r == 100 or r == -100:
                done = True
        rewards.append(r)
        total_hits.append(hits)
        total_wins.append(1 if r == 100 else 0)

    # print(f"Min player_vals value: {np.min(player_vals)}")
    # print(f"Max player_vals value: {np.max(player_vals)}")
    # print(f"Min ball_x_vals value: {np.min(ball_x_vals)}")
    # print(f"Max ball_x_vals value: {np.max(ball_x_vals)}")
    # print(f"Min ball_y_vals value: {np.min(ball_y_vals)}")
    # print(f"Max ball_y_vals value: {np.max(ball_y_vals)}")
    # print(f"Min vel_x_vals value: {np.min(vel_x_vals)}")
    # print(f"Max vel_x_vals value: {np.max(vel_x_vals)}")
    # print(f"Mean vel_x_vals value: {np.histogram(vel_x_vals)}")
    # print(f"Min vel_y_vals value: {np.min(vel_y_vals)}")
    # print(f"Max vel_y_vals value: {np.max(vel_y_vals)}")
    # print(f"Mean vel_y_vals value: {np.histogram(vel_y_vals)}")
    # print("Min Q value: ", np.min(Q))
    # print("Max Q value: ", np.max(Q))

    print(f"Wins: {np.sum(total_wins)}/1000")
    print(f"Average Hits: {np.mean(total_hits)}")
    print(f"Average Score: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
