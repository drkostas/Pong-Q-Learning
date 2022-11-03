"""
usage: python Q_Learning.py q a e n c f
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
    if len(args) == 6:
        # Load the arguments
        q, a, e, n, c, f = args
        # TODO:Cast the arguments to the correct type::Complete
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

    for i in range(num_train_episodes):
        agent.Run_Learning_Episode()
        if ((i) % 100 == 0):
            Progress_Bar((float(i))/num_train_episodes)
        if (i % check_freq == 0):
            vals = agent.Check()
            avg_score.append(vals[0])
            win_count.append(vals[1])

    Progress_Bar(1)
    agent.Save(file_name)

    print("Win Count:")
    print(win_count)
    print("Average Score:")
    print(avg_score)

# Progress bar to show progress


def Progress_Bar(progress):
    prog = "["
    for i in range(int(np.floor(progress*25))):
        prog = prog+"="
    for i in range(26-len(prog)):
        prog = prog+" "
    prog = prog+"]"
    print(prog)
    # Call some functions

# Training Agent


class Agent:
    def __init__(self, grid_dem, alpha, epsilon, map_size):
        self.grid_dem = grid_dem
        self.alpha = alpha
        self.epsilon = epsilon
        # Currently (player position, ball x, ball y)
        self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 3))
        self.map_size = map_size
        self.speed = 100

    def Run_Learning_Episode(self):
        p = pongGame(self.map_size, self.map_size, draw=False, totalSpeed=self.speed)

        done = False

        # Get init position info
        player, c, ball_x, ball_y = p.getState()[:4]
        player = self.Tab(player)
        ball_x = self.Tab(ball_x)
        ball_y = self.Tab(ball_y)

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
            player_i = self.Tab(player_i)
            ball_x_i = self.Tab(ball_x_i)
            ball_y_i = self.Tab(ball_y_i)

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

    # Check function that runs 500 episodes and recordes the average final reward and the number of wins

    def Check(self):
        reward = 0
        win_count = 0
        for i in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size, draw=False, totalSpeed=self.speed)
            done = False

            while (not done):

                # Get locations
                player, c, ball_x, ball_y = p.getState()[:4]
                player = self.Tab(player)
                ball_x = self.Tab(ball_x)
                ball_y = self.Tab(ball_y)

                # Choose Action
                action_values = self.Q[player, ball_x, ball_y]
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

    def Tab(self, item):
        val = int(np.floor((item/self.map_size)*self.grid_dem))
        if (val == self.grid_dem):
            val = val-1
        return val

    def Save(self, fileName):
        np.save(fileName, self.Q)


if __name__ == "__main__":
    main(sys.argv[1:])
