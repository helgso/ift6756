import copy

import gym
import numpy as np
import tensorflow as tf

from libs.shared import TargetMode

class GoGame():
    def __init__(self, board_size):        
        self.board_size = board_size
        self.gym = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='real')
        self.history = []

        # OpenAI variables
        self.state = None
        self.reward = None
        self.done = False
        self.info = None

    def clone(self):
        cloned_game = GoGame(self.board_size)
        cloned_game.history = copy.deepcopy(self.history)
        cloned_game.state = copy.deepcopy(self.state)
        cloned_game.reward = copy.deepcopy(self.reward)
        cloned_game.done = copy.deepcopy(self.done)
        cloned_game.info = copy.deepcopy(self.info)
        return cloned_game

    def is_done(self):
        return self.done
    
    def get_reward(self):
        return self.reward

    def get_action_size(self):
        return self.board_size**2

    def get_history(self):
        return self.history

    def to_play(self):
        return len(self.history) % 2

    def step(self, action):
        self.state, self.reward, self.done, self.info = self.gym.step(action)
        self.history.append(action)
        if len(self.get_legal_actions()) == 0:
            self.end_game()
        return self.state, self.reward, self.done, self.info
    
    def reset(self):
        self.history = []
        self.state = self.gym.reset()
        self.reward = None
        self.done = False
        self.info = None
        return self.state

    def get_board_state(self):
        if self.state is None:
            return np.zeros((self.board_size, self.board_size, 1))
        current_player = self.to_play()
        current_player_pieces = self.state[current_player]
        opponent_player_pieces = self.state[1-current_player]
        board = current_player_pieces - opponent_player_pieces
        return np.expand_dims(board, axis=2)

    def end_game(self):
        # Passing twice ends the game
        self.gym.step(None)
        self.state, self.reward, self.done, self.info = self.gym.step(None)

    def get_z_values(self):
        n_rounds = len(self.history)
        if self.reward == 0: # Tie. Punish both sides
            values = np.resize([-1.0], n_rounds)
        elif self.reward == 1: # Black won
            values = np.resize([1.0, -1.0], n_rounds)
        elif self.reward == -1: # White won
            values = np.resize([-1.0, 1.0], n_rounds)
        return np.expand_dims(values, axis=1)

    def get_legal_actions(self):
        if self.info == None:
            return np.arange(self.board_size**2)
        # -1 because the last entry is whether it is forbidden to pass or not
        return np.where(self.info['invalid_moves'][:-1] == 0)[0]
