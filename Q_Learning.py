"""
usage: python Q_Learning.py grid_dem alpha epsilon num_episodes check_freq file_name [agent_type]
description: Runs the Q-Learning algorithm on a grid world
output: Saves the Q-table to a file
"""

import sys
import numpy as np
from agents import Agent, Agent_DL


MAP_SIZE = 300
GAME_SPEED = 2
DRAW = False
INCLUDE_VEL = False


def load_args():
    """ Loads the arguments from the command line"""
    args = sys.argv[1:]
    # Load the arguments
    if len(args) == 6:
        grid_dem, alpha, epsilon, num_episodes, check_freq, file_name = args
        agent_type = 'classic'
    elif len(args) == 7:
        grid_dem, alpha, epsilon, num_episodes, check_freq, file_name, agent_type = args
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    # Cast all arguments and make sure they are proper data types
    try:
        grid_dem = int(grid_dem)
        alpha = float(alpha)
        epsilon = float(epsilon)
        num_episodes = int(num_episodes)
        check_freq = int(check_freq)
        file_name = str(file_name)
    except ValueError:
        raise Exception("Invalid Parameter Types")

    return grid_dem, alpha, epsilon, num_episodes, check_freq, file_name, agent_type


def clear_line():
    """ Clears the last line in the terminal"""
    print("\033[A                             \033[A")


def progress_bar(progress: float) -> None:
    """ Custom Progress bar to show progress of training."""
    clear_line()
    prog = "["
    for _ in range(int(np.floor(progress*25))):
        prog += "="
    prog += ">"
    for _ in range(26-len(prog)):
        prog += " "
    prog = prog+"]"+str(int(np.floor(progress*100)))+"%"
    print(prog)


def main():
    """ Main function"""
    # --- Args Loading and Error Checking --- #
    print(f"{' Initializing ':-^30}")
    (grid_dem, alpha, epsilon, num_episodes,
     check_freq, file_name, agent_type) = load_args()
    save_path = "Q_tables/"+file_name

    # -- Initialize the agent -- #
    if agent_type == 'DL':
        print()
        agent = Agent_DL(alpha=alpha, epsilon=epsilon, game_speed=GAME_SPEED,
                         render_game=DRAW, map_size=MAP_SIZE,
                         include_vel=INCLUDE_VEL)
    else:  # Defaults to `classic` agent
        agent = Agent(grid_dem=grid_dem, alpha=alpha, epsilon=epsilon,
                      map_size=MAP_SIZE, game_speed=GAME_SPEED,
                      render_game=DRAW, include_vel=INCLUDE_VEL)

    # --- Training --- #
    clear_line()
    print(f"{' Training Starts ':-^30}")
    print("Progress:\n")
    win_count = []
    avg_score = []

    # @GCantrall: alternatively we could use tqdm here:
    # from tqdm import tqdm
    # for i in tqdm(range(num_episodes)):
    for i in range(num_episodes):
        agent.run_learning_episode()
        progress_bar((float(i))/num_episodes)
        if (i % check_freq == 0):
            vals = agent.check()
            avg_score.append(vals[0])
            win_count.append(vals[1])
            # print(str((agent.Q==0).sum())+"/"+str(agent.Q.size))

    progress_bar(1)

    print("Win Count:")
    print(f"{int((wins_np:=np.array(win_count)).mean())}/{wins_np.shape[0]}")
    print("Average Score:")
    print(np.array(avg_score).mean())
    agent.save(save_path)


if __name__ == "__main__":
    main()
