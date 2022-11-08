from typing import *
from abc import ABC, abstractmethod
import os
import numpy as np
import random
import pickle
from pong.pongclass import pongGame


class AbstractAgent(ABC):
    """ Abstract Agent Class"""

    def __init__(self, alpha: float, epsilon: float,
                 map_size: int, include_vel: bool, 
                 game_speed: float, render_game: bool, 
                 grid_dem: int = None):
        self.grid_dem = grid_dem
        self.alpha = alpha
        self.epsilon = epsilon
        self.include_vel = include_vel
        self.map_size = map_size
        self.game_speed = game_speed
        self.render_game = render_game

    def save(self, file_name: str) -> None:
        out = {'alpha': self.alpha, 'epsilon': self.epsilon,
               'Q': self.Q, 'grid_dem': self.grid_dem,
               'map_size': self.map_size,
               'include_vel': self.include_vel}
        # Save as pickle
        with open(file_name+'.pkl', 'wb') as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
        print(f"Training Data saved to `{file_name}.pkl`")

    @abstractmethod
    def run_learning_episode(self, game_speed: float, render_game: bool) -> None:
        pass

    @abstractmethod
    def check(self) -> Tuple[float, int]:
        pass


class Agent(AbstractAgent):
    """ Agent for playing a game of pong. """

    def __init__(self, grid_dem: int, alpha: float, epsilon: float,
                 map_size: int, game_speed: float, 
                 render_game: bool, include_vel: bool = True):
        # Initialize Abstract Class
        super().__init__(alpha=alpha, epsilon=epsilon, map_size=map_size, 
                         include_vel=include_vel, game_speed=game_speed, 
                         render_game=render_game, grid_dem=grid_dem)
        # Initialize Q Table
        if self.include_vel:
            # (player position, ball x, ball y, velocity x, velocity y)
            self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 2, 4, 3))*2
        else:
            # (player position, ball x, ball y, velocity x, velocity y)
            self.Q = np.zeros((grid_dem, grid_dem, grid_dem, 3))*2

    def run_learning_episode(self) -> None:
        p = pongGame(self.map_size, self.map_size,
                     game_speed=self.game_speed, draw=self.render_game)
        done = False

        # Get init position info
        player, _, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]
        player = self.tab(player)
        ball_x = self.tab(ball_x)
        ball_y = self.tab(ball_y)

        if (self.include_vel):
            vel_x = self.tab_vel_x(vel_x)
            vel_y = self.tab_vel_y(vel_y)
            
        while not done:
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
            state = p.getState()[:6]
            player_i, c, ball_x_i, ball_y_i, vel_x_i, vel_y_i = state
            player_i = self.tab(player_i)
            ball_x_i = self.tab(ball_x_i)
            ball_y_i = self.tab(ball_y_i)

            if (self.include_vel):
                vel_x_i = self.tab_vel_x(vel_x_i)
                vel_y_i = self.tab_vel_y(vel_y_i)

            # Update Q Value
            if self.include_vel:
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

    def check(self) -> Tuple[float, int]:
        """ check function that runs 500 episodes and recordes 
            the average final reward and the number of wins. """
        reward = 0
        win_count = 0
        for _ in range(500):

            # Create Pong game object
            p = pongGame(self.map_size, self.map_size,
                         game_speed=self.game_speed, draw=self.render_game)
            done = False

            while (not done):

                # Get locations
                state = p.getState()[:6]
                player, _, ball_x, ball_y, vel_x, vel_y = state
                player = self.tab(player)
                ball_x = self.tab(ball_x)
                ball_y = self.tab(ball_y)

                if self.include_vel:
                    vel_x = self.tab_vel_x(vel_x)
                    vel_y = self.tab_vel_y(vel_y)

                # Choose Action
                if self.include_vel:
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

    def tab(self, item: float) -> int:
        val = int(np.floor((item/self.map_size)*self.grid_dem))
        if (val >= self.grid_dem):
            val = self.grid_dem-1
        return val

    def tab_vel_x(self, vel: float) -> int:
        if (vel < 0):
            return 0
        else:
            return 1

    def tab_vel_y(self, vel: float) -> int:
        if (vel < -.5):
            return 0
        elif (vel < 0):
            return 1
        elif (vel < .5):
            return 2
        else:
            return 3


class Agent_DL(AbstractAgent):
    """ Deep Learning Agent for playing a game of pong. """

    def __init__(self, alpha: float, epsilon, map_size, 
                 game_speed: float, render_game: bool, 
                 include_vel=True):
        # Initialize Abstract Class
        super().__init__(alpha=alpha, epsilon=epsilon, map_size=map_size, 
                         include_vel=include_vel, game_speed=game_speed, 
                         render_game=render_game)
        # Load it here so that you can run the other
        # agents without installing tensorflow
        os.environ['AUTOGRAPH_VERBOSITY'] = '0'
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        
        # Initialize Q Table
        self.Q = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(6,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.Q.compile(optimizer='adam',
                       loss='MSE',
                       metrics=['accuracy', 'mae'])
        print(self.Q.summary())

    def run_learning_episode(self):
        p = pongGame(self.map_size, self.map_size,
                     game_speed=self.game_speed, draw=self.render_game)

        done = False

        # Get init position info
        player, c, ball_x, ball_y, vel_x, vel_y = p.getState()[:6]

        while (not done):

            # Choose Action
            action_values = []
            for i in range(3):
                x = np.array([player, c, ball_x, ball_y, vel_x, vel_y])
                action_value = self.Q.predict(x=x.reshape(1, 6),
                                              verbose=0)[0][0]
                action_values.append(action_value)

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
            for _ in range(3):
                x = np.array([player, c, ball_x, ball_y, vel_x, vel_y])
                action_value = self.Q.predict(x=x.reshape(1, 6),
                                              verbose=0)[0][0]
                future_actions.append(action_value)

            expected_val = action_values[action] +\
                self.alpha*(r+max(future_actions) -
                            action_values[action])

            x = np.array([player, c, ball_x, ball_y, vel_x, vel_y])
            y = np.array([expected_val])
            self.Q.fit(x=x.reshape(1, 6), y=y.reshape(1, 1),
                       verbose=0)

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
                    x = np.array([player, c, ball_x, ball_y, vel_x, vel_y])
                    action_value = self.Q.predict(x=x.reshape(1, 6), verbose=0)
                    action_values.append(action_value)

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
