import gym
from gym import spaces
#import pygame
import numpy as np


class Env2048(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, high = 4, wide = 4, seed = None, invalid_move_warmup=16,
                 invalid_move_threshold=0.1, penalty=-32,):

        self.__high = high
        self.__wide = wide

        self.__invalid_move_warmup = invalid_move_warmup
        self.__invalid_move_threshold = invalid_move_threshold
        self.__penalty = penalty

        self.action_space = spaces.Discrete(4)

        self.__score = 0
        self.__total_score = 0
        self.__step = 0
        self.__invalid_step = 0
        self.__max_block = 0
        self.__no_rest = False
        self.__deep_ob = int(np.log2(2048))

        self.__board = np.zeros((self.__high, self.__wide), dtype=np.int32)
        self.__temp_board = np.zeros((self.__high, self.__wide), dtype=np.int32)
        self.observation_space = spaces.Box(0, 1, (self.__high, self.__wide, self.__deep_ob), dtype=np.int32)
        self.__add_block()
        self.__add_block()

        self.window = None
        self.clock = None

    def __add_block(self):
        '''This is used to add new blocks to the rest of the board.'''
        try:
            rest = np.argwhere(self.__board == 0)
            choose = np.random.choice(len(rest))
            self.__board[rest[choose][0],rest[choose][1]] = 2
        except:
            self.__no_rest = True

    def __move(self, action):
        '''This part of the code is used to apply move
        0: move up, 1: move down, 2: move left, 3: move right.'''
        self.__score = 0
        if action == 0:
            self.__up()
        elif action == 1:
            self.__dowm()
        elif action == 2:
            self.__left()
        elif action == 3:
            self.__right()

    def __up(self):
        for row in self.__temp_board.T:
            self.__move_zore(row)
            self.__merge(row)
            self.__move_zore(row)
        return None
    def __dowm(self):
        temp_board = np.flipud(self.__temp_board).copy()
        for row in temp_board.T:
            self.__move_zore(row)
            self.__merge(row)
            self.__move_zore(row)
        self.__temp_board = np.flipud(temp_board).copy()
        return None
    def __left(self):
        for row in self.__temp_board:
            self.__move_zore(row)
            self.__merge(row)
            self.__move_zore(row)
        return None
    def __right(self):
        temp_board = np.fliplr(self.__temp_board).copy()
        for row in temp_board:
            self.__move_zore(row)
            self.__merge(row)
            self.__move_zore(row)
        self.__temp_board = np.fliplr(temp_board).copy()
        return None

    def __move_zore(self, row):
        '''This part is moved 0 value to the end for each row and each direction.'''
        flag = 0
        for i in row:
            if i:
                row[flag] = i
                flag = flag + 1
        for i in range(flag, len(row)):
            row[i] = 0
    def __merge(self, row):
        '''This part is to merge the same value if they are adjacent.'''
        for temp in range(1, len(row)):
            if (row[temp - 1] == row[temp]):
                row[temp - 1] = row[temp - 1] * 2
                self.__score += row[temp - 1]
                row[temp] = 0
                if(self.__max_block < row[temp - 1]):
                    self.__max_block = row[temp - 1]

    def __check_move(self):
        '''This part is to check whether the move is valid or not.'''
        self.__step += 1
        self.__total_score += self.__score
        if np.array_equal(self.__temp_board, self.__board):
            self.__invalid_step += 1
            self.__score = self.__penalty
        else:
            self.__board = self.__temp_board.copy()
            self.__add_block()

    def __check_state(self):
        '''This part is to check whether the game is ending or not.'''

        '''For invalid move'''
        if(
            self.__invalid_step > self.__invalid_move_warmup and
            self.__invalid_step > self.__invalid_move_threshold * self.__step
        ):
            return True, self.__score + self.__penalty
        elif (self.__max_block == 2048):
            '''check whether the max value is get the 2048 or not'''
            return True, self.__score + 2048
        elif(self.__no_rest):
            for row in self.__board:
                for temp in range(1, len(row)):
                    if (row[temp - 1] == row[temp]):
                        return False, self.__score
            for row in self.__board.T:
                for temp in range(1, len(row)):
                    if (row[temp - 1] == row[temp]):
                        return False, self.__score
        return False, self.__score


    def __create_ob(self):
        temp_ob = np.zeros((self.__high, self.__wide, self.__deep_ob))
        for h in range(self.__high):
            for w in range(self.__wide):
                if self.__board[h,w]:
                    temp_ob[h,w,(int(np.log2(self.__board[h,w]))-1)] = 1
        return temp_ob


    def step(self, action):
        info = dict()
        self.__temp_board = self.__board.copy()
        info["old_state"]=self.__board.copy()
        self.__move(action)
        self.__check_move()
        done, reward = self.__check_state()
        info["step"] = self.__step
        info["new_state"] = self.__board.copy()
        info["total_score"] = self.__total_score
        info["max_block"] = self.__max_block
        observation  = self.__create_ob()
        return(observation, float(reward), done, info)

    def reset(self):
        self.__score = 0
        self.__total_score = 0
        self.__step = 0
        self.__invalid_step = 0
        self.__max_block = 0
        self.__no_rest = False

        self.__board = np.zeros((self.__high, self.__wide), dtype=np.int32)
        self.__temp_board = np.zeros((self.__high, self.__wide), dtype=np.int32)
        self.__add_block()
        self.__add_block()

        observation = self.__create_ob()
        return observation

    def render(self, mode="human"):
        pass


    def set_boaed(self, board):
        self.__board = board