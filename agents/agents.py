from pong.pongclass import pongGame
import numpy as np
import random


class Agent:
    """ Agent for playing a game of pong """

    def __init__(self, grid_dem, alpha, epsilon, map_size,
                 include_vel=True):
        self.grid_dem = grid_dem
        self.alpha = alpha
        self.epsilon = epsilon
        self.include_vel = include_vel
        if (self.include_vel):
            # Currently (player position, ball x, ball y, velocity x, velocity y)
            self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 2, 4, 3))
        else:
            # Currently (player position, ball x, ball y, velocity x, velocity y)
            self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 3))
        self.map_size = map_size

    def run_learning_episode(self):
        p = pongGame(self.map_size, self.map_size, draw=False, game_speed=100)

        done = False

        # Get init position info
        player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]
        player = self.tab(player)
        ball_x = self.tab(ball_x)
        ball_y = self.tab(ball_y)

        if (self.include_vel):
            vel_x = self.tab_vel_x(vel_x)
            vel_y = self.tab_vel_y(vel_y)

        while (not done):

            # Choose Action
            if (self.include_vel):
                action_values = self.Q[player, ball_x, ball_y, vel_x, vel_y]
            else:
                action_values = self.Q[player, ball_x, ball_y]

            ran = random.random()
            if (ran > self.epsilon):
                action = np.argmax(action_values)
            else:
                action = random.randint(0, 2)

            # Take Action
            r = p.takeAction(action)

            # Determine new state
            player_i, c, ball_x_i, ball_y_i, vel_x_i, vel_y_i = p.getState()[
                :6]
            player_i = self.tab(player_i)
            ball_x_i = self.tab(ball_x_i)
            ball_y_i = self.tab(ball_y_i)

            if (self.include_vel):
                vel_x_i = self.tab_vel_x(vel_x_i)
                vel_y_i = self.tab_vel_y(vel_y_i)

            # Update Q Value
            if (self.include_vel):
                self.Q[player, ball_x, ball_y, vel_x, vel_y, action] = self.Q[player, ball_x, ball_y, vel_x, vel_y, action] +\
                    self.alpha*(r+max(self.Q[player_i, ball_x_i, ball_y_i, vel_x_i, vel_y_i]) -
                                self.Q[player, ball_x, ball_y, vel_x, vel_y, action])
            else:
                self.Q[player, ball_x, ball_y, action] = self.Q[player, ball_x, ball_y, action] + \
                    self.alpha * (r + max(self.Q[player_i, ball_x_i, ball_y_i]) -
                                  self.Q[player, ball_x, ball_y, action])

            # Update State values
            player = player_i
            ball_x = ball_x_i
            ball_y = ball_y_i
            if (self.include_vel):
                vel_x = vel_x_i
                vel_y = vel_y_i

            if (r == 100 or r == -100):
                done = True

    def check(self):
        """ check function that runs 500 episodes and recordes 
            the average final reward and the number of wins. """
        reward = 0
        win_count = 0
        for i in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size)
            done = False

            while (not done):

                # Get locations
                player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]
                player = self.tab(player)
                ball_x = self.tab(ball_x)
                ball_y = self.tab(ball_y)

                if (self.include_vel):
                    vel_x = self.tab_vel_x(vel_x)
                    vel_y = self.tab_vel_y(vel_y)

                # Choose Action
                if (self.include_vel):
                    action_values = self.Q[player,
                                           ball_x, ball_y, vel_x, vel_y]
                else:
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

    def tab(self, item):
        val = int(np.floor((item/self.map_size)*self.grid_dem))
        if (val >= self.grid_dem):
            val = self.grid_dem-1
        return val

    def tab_vel_x(self, vel):
        if (vel < 0):
            return 0
        else:
            return 1

    def tab_vel_y(self, vel):
        if (vel < -.5):
            return 0
        elif (vel < 0):
            return 1
        elif (vel < .5):
            return 2
        else:
            return 3

    def save(self, file_name):
        np.save(file_name, self.Q)


class Agent_DL:

    def __init__(self,  alpha, epsilon, map_size,
                 include_vel=True):
        # Load it here so that you can run the other
        # agents without installing tensorflow
        import tensorflow as tf
        self.alpha = alpha
        self.epsilon = epsilon
        self.map_size = map_size
        self.include_vel = include_vel
        self.Q = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(6)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.Q.compile(optimizer='adam',
                       loss='MSE',
                       metrics=['accuracy'])
        print(self.Q.summary())

    def run_learning_episode(self):
        p = pongGame(self.map_size, self.map_size, draw=False, game_speed=100)

        done = False

        # Get init position info
        player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]

        while (not done):

            # Choose Action
            action_values = []
            for i in range(3):
                action_values.append(self.Q.predict(
                    [[player, ball_x, ball_y, vel_x, vel_y, i]], verbose=0)[0][0])

            ran = random.random()
            if (ran > self.epsilon):
                action = np.argmax(action_values)
            else:
                action = random.randint(0, 2)

            # Take Action
            r = p.takeAction(action)

            # Determine new state
            state = p.getState()[:6]
            player_i, _, ball_x_i, ball_y_i, vel_x_i, vel_y_i = state

            future_actions = []
            for i in range(3):
                future_actions.append(self.Q.predict(
                    [[player_i, ball_x_i, ball_y_i, vel_x_i, vel_y_i, i]], verbose=0)[0][0])

            expected_val = action_values[action] +\
                self.alpha*(r+max(future_actions) -
                            action_values[action])

            self.Q.fit([[player, ball_x, ball_y, vel_x, vel_y, action]], [
                       [expected_val]], verbose=0)

            # Update State values
            player = player_i
            ball_x = ball_x_i
            ball_y = ball_y_i

            vel_x = vel_x_i
            vel_y = vel_y_i

            if (r == 100 or r == -100):
                done = True

    def check(self):
        """ check function that runs 500 episodes and recordes 
            the average final reward and the number of wins. """
        reward = 0
        win_count = 0
        for i in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size)
            done = False
            player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]
            while (not done):

                # Get locations
                # Choose Action
                action_values = []
                for i in range(3):
                    action_values.append(self.Q.predict(
                        [[player, ball_x, ball_y, vel_x, vel_y, i]], verbose=0))

                ran = random.random()
                if (ran > self.epsilon):
                    action = np.argmax(action_values)
                else:
                    action = random.randint(0, 2)

                # Take Action
                r = p.takeAction(action)
                reward = reward + r

                if (r == 100 or r == -100):
                    done = True

                    if (r == 100):
                        win_count = win_count+1
        reward = reward/500
        return reward, win_count

    def save(self, file_name):
        np.save(file_name, self.Q)
