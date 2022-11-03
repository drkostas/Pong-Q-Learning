"""
usage: python Q_Learning.py q a e n c f
"""
import random
import sys
import time
import numpy as np
from pong.pongclass import pongGame
# from tqdm import tqdm


class Agent:
    """ Agent for playing a game of pong """

    def __init__(self, grid_dem: int, alpha: float, epsilon: float, 
                 map_size: int = 300, speed: int = 100):
        self.grid_dem = grid_dem
        self.alpha = alpha
        self.epsilon = epsilon
        # Currently (player position, ball x, ball y)
        self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 3))
        self.map_size = map_size
        self.speed = speed

    def run_learning_episode(self):
        p = pongGame(self.map_size, self.map_size,
                     draw=False, totalSpeed=self.speed)

        done = False

        # Get init position info
        player, c, ball_x, ball_y = p.getState()[:4]
        player = self.tab(player)
        ball_x = self.tab(ball_x)
        ball_y = self.tab(ball_y)

        while (not done):

            # Choose Action
            action_values = self.Q[player, ball_x, ball_y]

            ran = random.random()
            if (ran > self.epsilon):
                action = np.argmax(action_values)
            else:
                action = random.randint(0, 2)

            # Take Action
            r = p.takeAction(action)

            # Determine new state
            player_i, c, ball_x_i, ball_y_i = p.getState()[:4]
            player_i = self.tab(player_i)
            ball_x_i = self.tab(ball_x_i)
            ball_y_i = self.tab(ball_y_i)

            # Update Q Value
            self.Q[player, ball_x, ball_y, action] = self.Q[player, ball_x, ball_y, action] +\
                self.alpha*(r+max(self.Q[player_i, ball_x_i, ball_y_i]) -
                            self.Q[player, ball_x, ball_y, action])

            # Update State values
            player = player_i
            ball_x = ball_x_i
            ball_y = ball_y_i

            if (r == 100 or r == -100):
                done = True


    def check(self):
        """ Check function that runs 500 episodes and recordes the 
            average final reward and the number of wins. """
        reward = 0
        win_count = 0
        for i in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size,
                         draw=False, totalSpeed=self.speed)
            done = False

            while (not done):

                # Get locations
                player, c, ball_x, ball_y = p.getState()[:4]
                player = self.tab(player)
                ball_x = self.tab(ball_x)
                ball_y = self.tab(ball_y)

                # Choose Action
                try:
                    action_values = self.Q[player, ball_x, ball_y]
                except Exception as e:
                    print("Tried to retrieve Q value with:",
                          (player, ball_x, ball_y))
                    print("Q shape:", self.Q.shape)
                    raise e
                action = np.argmax(action_values)

                # Take action and observe reward
                r = p.takeAction(action)
                reward = reward + r

                if (r == 100 or r == -100):
                    done = True

                    if (r == 100):
                        win_count = win_count+1
        reward = reward/500
        return reward, win_count

    def tab(self, item):
        val = int(np.floor((item/self.map_size)*self.grid_dem))
        if (val == self.grid_dem):
            val = val-1
        return val

    def save(self, file_name):
        np.save(file_name, self.Q)


def progress_bar(progress: float) -> None:
    """ Custom Progress bar to show progress of training."""
    prog = "["
    for _ in range(int(np.floor(progress*25))):
        prog = prog+"="
    for _ in range(26-len(prog)):
        prog = prog+" "
    prog = prog+"]"
    print(prog)


def main(args):
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    if len(args) == 6:
        # Load the arguments
        q, a, e, n, c, f = args
    else:
        # If more or less arguments are given, print an error message
        raise Exception("Invalid number of arguments")
    # Cast all arguments and make sure they are proper data types
    try:
        grid_dem = int(q)
        alpha = float(a)
        epsilon = float(e)
        num_train_episodes = int(n)
        check_freq = int(c)
        file_name = str(f)
    except ValueError:
        raise Exception("Invalid Parameter Types")

    agent = Agent(grid_dem, alpha, epsilon, 300)

    print("Training Progress:")

    win_count = []
    avg_score = []

    # @GCantrall: alternatively we could use tqdm here:
    # from tqdm import tqdm
    # for i in tqdm(range(num_train_episodes)):
    for i in range(num_train_episodes):
        agent.run_learning_episode()
        if ((i) % 100 == 0):
            progress_bar((float(i))/num_train_episodes)
        if (i % check_freq == 0) and False:
            vals = agent.check()
            avg_score.append(vals[0])
            win_count.append(vals[1])

    progress_bar(1)
    agent.save(file_name)

    print("Win Count:")
    print(win_count)
    print("Average Score:")
    print(avg_score)


if __name__ == "__main__":
    main(sys.argv[1:])
