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
        if (i % check_freq == 0):
            vals = agent.check()
            avg_score.append(vals[0])
            win_count.append(vals[1])
            print(str((agent.Q==0).sum())+"/"+str(agent.Q.size))

    progress_bar(1)
    agent.save(file_name)

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
        self.include_Vel=True
        if(self.include_Vel):
            self.Q = np.zeros((grid_dem,grid_dem,grid_dem,2,4,3)) # Currently (player position, ball x, ball y, velocity x, velocity y)
        else:
            self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 3))   # Currently (player position, ball x, ball y, velocity x, velocity y)
        self.map_size = map_size



    def Run_Learning_Episode(self):
        p = pongGame(self.map_size,self.map_size)

        done = False

        # Get init position info
        player,c , ball_x, ball_y,vel_x,vel_y = p.getState()[:6]
        player = self.Tab(player)
        ball_x = self.Tab(ball_x)
        ball_y = self.Tab(ball_y)

        if(self.include_Vel):
            vel_x = self.Tab_Vel_x(vel_x)
            vel_y = self.Tab_Vel_y(vel_y)

        while (not done):

            # Choose Action
            if(self.include_Vel):
                action_values = self.Q[player, ball_x, ball_y, vel_x,vel_y]
            else:
                action_values = self.Q[player,ball_x,ball_y]

            ran  = random.random()
            if(ran >self.epsilon):
                action = np.argmax(action_values)
            else:
                action = random.randint(0,2)

            # Take Action
            r = p.takeAction(action)

            # Determine new state
            player_i, c,ball_x_i, ball_y_i, vel_x_i, vel_y_i = p.getState()[:6]
            player_i = self.Tab(player_i)
            ball_x_i = self.Tab(ball_x_i)
            ball_y_i = self.Tab(ball_y_i)

            if (self.include_Vel):
                vel_x_i = self.Tab_Vel_x(vel_x_i)
                vel_y_i = self.Tab_Vel_y(vel_y_i)
            # Update Q Value

            if(self.include_Vel):
                self.Q[player,ball_x,ball_y,vel_x,vel_y, action] = self.Q[player, ball_x, ball_y, vel_x, vel_y, action]+\
                    self.alpha*(r+max(self.Q[player_i,ball_x_i,ball_y_i,vel_x_i, vel_y_i])- \
                    self.Q[player, ball_x, ball_y,vel_x, vel_y, action])
            else:
                self.Q[player, ball_x, ball_y, action] = self.Q[player, ball_x, ball_y, action] + \
                    self.alpha * (r + max(self.Q[player_i, ball_x_i, ball_y_i]) - \
                    self.Q[player, ball_x, ball_y, action])

            # Update State values
            player = player_i
            ball_x = ball_x_i
            ball_y = ball_y_i
            if(self.include_Vel):
                vel_x = vel_x_i
                vel_y = vel_y_i

            if (r == 100 or r == -100):
                done = True

    # Check function that runs 500 episodes and recordes the average final reward and the number of wins

    def Check(self):
        reward = 0
        win_count =0
        for i in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size)
            done = False

            while (not done):

                # Get locations
                player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]
                player = self.Tab(player)
                ball_x = self.Tab(ball_x)
                ball_y = self.Tab(ball_y)

                if (self.include_Vel):
                    vel_x = self.Tab_Vel_x(vel_x)
                    vel_y = self.Tab_Vel_y(vel_y)


                # Choose Action
                if (self.include_Vel):
                    action_values = self.Q[player, ball_x, ball_y, vel_x, vel_y]
                else:
                    action_values = self.Q[player, ball_x, ball_y]

                action = np.argmax(action_values)

                # Take action and observe reward
                r = p.takeAction(action)
                reward = reward +r

                if (r == 100 or r == -100):
                    done = True

                    if(r == 100):
                        win_count = win_count+1
        reward = reward/500
        return reward,win_count




    def Tab(self, item):
        val = int(np.floor((item/self.map_size)*self.grid_dem))
        if(val>=self.grid_dem):
            val = self.grid_dem-1
        return val

    def Tab_Vel_x(self,vel):
        if(vel<0):
            return 0
        else:
            return 1


    def Tab_Vel_y(self,vel):
        if(vel<-.5):
            return 0
        elif(vel<0):
            return 1
        elif(vel<.5):
            return 2
        else:
            return 3

    def Save(self, fileName):
        np.save(fileName,self.Q)


if __name__ == "__main__":
    main(sys.argv[1:])
