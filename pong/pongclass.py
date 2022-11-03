import pygame
import math
import numpy as np
import random


# pong game class
class pongGame:
    # initializing parameters
    def __init__(self, h, w, draw=True, totalSpeed=2):
        # window height
        self.h = h
        # window width
        self.w = w
        # If you intend to use for visualization, set draw to True
        if (draw):
            pygame.init()
            self.window = pygame.display.set_mode((self.h, self.w))
        else:
            self.window = None
        # ball x and y location
        self.xball = self.w/2
        self.yball = self.h/2
        # ball speed and angle
        self.angle = random.random()*0.5*math.pi+0.75*math.pi
        self.totalSpeed = totalSpeed
        self.ballHDirection = self.totalSpeed*math.cos(self.angle)
        self.ballVDirection = self.totalSpeed*math.sin(self.angle)
        # player paddle location
        self.y1 = self.h/2-40
        # computer paddle location
        self.y2 = self.h/2
        # paddle length
        self.paddle_length = self.h/6

    # returns all the parameters for the game (player location, computer location, x of ball, y of ball, x direction of ball, y direction of ball)

    def getState(self):
        return np.array([self.y1, self.y2, self.xball, self.yball, self.ballHDirection, self.ballVDirection])

    # Take one step of the game
    def takeAction(self, action):
        # reward
        r = 0
        # move action (up is 0, down is 1, no move is 2)
        if (action == 0 and self.y1 > 5):
            self.y1 = self.y1-5
        elif (action == 1 and self.y1 < self.w-5):
            self.y1 = self.y1+5

        # move computer paddle on its own
        if (self.yball > self.y2+self.paddle_length/2):
            self.y2 = self.y2+5
        elif (self.yball < self.y2+self.paddle_length/2):
            self.y2 = self.y2-5

        # math for when paddle hits ball
        if (self.yball > self.y1 and self.yball < self.y1+self.paddle_length and self.xball > 0 and self.xball < 15):
            self.totalSpeed = self.totalSpeed+0.2
            self.angle = (math.pi/4)*(self.yball-(self.y1 +
                                                  self.paddle_length/2))/(self.paddle_length/2)
            self.xball = 15
            r = 1
        elif (self.yball > self.y2 and self.yball < self.y2+self.paddle_length and self.xball > self.w-15 and self.xball < self.w):
            self.totalSpeed = self.totalSpeed+0.2
            self.angle = math.pi - \
                (math.pi/4)*(self.yball-(self.y2+self.paddle_length/2)) / \
                (self.paddle_length/2)
            self.xball = self.w-15

        # if you lose
        if (self.xball < 0):
            r = -100
        # if you win
        elif (self.xball > self.w):
            r = 100
        # if ball hits top or bottom wall
        if (self.yball <= 0 or self.yball >= self.h):
            self.angle = -self.angle

        # recalculate ball location
        self.ballHDirection = self.totalSpeed*math.cos(self.angle)
        self.ballVDirection = self.totalSpeed*math.sin(self.angle)
        self.xball = self.xball+self.ballHDirection
        self.yball = self.yball+self.ballVDirection

        # return reward
        return r

    # a function to draw the actual game frame. If called it is probably best to use a delay between frames (for example time.sleep(0.03) for about 30 frames per second.
    def draw(self):
        # clear the display
        self.window.fill(0)

        # draw the scene
        pygame.draw.rect(self.window, (255, 255, 255),
                         (5, self.y1, 10, self.paddle_length))
        pygame.draw.rect(self.window, (255, 255, 255),
                         (self.w-15, self.y2, 10, self.paddle_length))
        pygame.draw.circle(self.window, (255, 255, 255),
                           (self.xball, self.yball), 5)
        # update the display
        pygame.display.flip()
