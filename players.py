import itertools
import random
import numpy as np
import nn


class QPlayer:
    def __init__(self, gamma, net_lr = 0.01):
        self.neur_net = nn.NN([64, 128, 128, 64, 64], net_lr)
        self.explore_rate = 0.6
        self.gamma = gamma
        self.game_list = []
        self.wins = 0

    def play(self, place_func, board_state, me, save = True):
        y = np.apply_along_axis(lambda x: int((x == me and 1) or (x != 0 and -1)), 1, board_state.reshape((64, 1))).reshape((64, 1)) #the board
        comp = False
        pos = None
        if np.random.random() < self.explore_rate:
            x = []
            for i in range(8):
                for j in range(8):
                    x.append((i, j))
            random.shuffle(x)
            while not comp and x:
                pos = x.pop()
                comp = place_func(*pos)

            if not comp and not x:
                return False

        else:
            output = self.neur_net.forward_pass(y)
            x = [(v,i) for i,v in enumerate(output)]
            x.sort(key = lambda x: x[0], reverse = True)
            while not comp and x:
                next_move_loc = x.pop()[1]
                pos = int(next_move_loc / 8), next_move_loc % 8
                comp = place_func(*pos)
            if not comp and not x:
                return False

        if save is True:
            self.game_list.append((np.copy(y), pos[0] * 8 + pos[1]))

        return True

    def weight_update(self, score):
        i = 0
        state, action = self.game_list[i]
        q = self.neur_net.forward_pass(state)
        for i in range(len(self.game_list)):
            i += 1
            if i == len(self.game_list):
                q[action] = score

            else:
                next_state, next_action = self.game_list[i]
                updated_q = self.neur_net.forward_pass(next_state)
                q[action] += self.gamma * np.max(updated_q)
            self.neur_net.backProp(state, self.neur_net.mkVec(q))
            if i != len(self.game_list):
                action, q = next_action, updated_q

class SarsaPlayer:
    def __init__(self, gamma, net_lr = 0.01):
        self.neur_net = nn.NN([64, 128, 128, 64, 64], net_lr)
        self.explore_rate = 0.6
        self.gamma = gamma
        self.game_list = []
        self.wins = 0

    def play(self, place_func, board_state, me, save = True):
        y = np.apply_along_axis(lambda x: int((x == me and 1) or (x!=0 and -1)), 1, board_state.reshape((64, 1))).reshape((64, 1)) #the board
        comp = False
        pos = None
        if np.random.random() < self.explore_rate:
            x = []
            for i in range(8):
                for j in range(8):
                    x.append((i, j))
            random.shuffle(x)
            while not comp and x:
                pos = x.pop()
                comp = place_func(*pos)

            if not comp and not x:
                return False

        else:
            output = self.neur_net.forward_pass(y)
            x = [(v,i) for i,v in enumerate(output)]
            x.sort(key = lambda x: x[0], reverse = True)
            while not comp and x:
                next_move_loc = x.pop()[1]
                pos = int(next_move_loc / 8), next_move_loc % 8
                comp = place_func(*pos)
            if not comp and not x:
                return False

        if save is True:
            self.game_list.append((np.copy(y), pos[0] * 8 + pos[1]))

        return True

    def weight_update(self, score):
        i = 0
        state, action = self.game_list[i]
        q = self.neur_net.forward_pass(state)
        for i in range(len(self.game_list)):
            i += 1
            if i == len(self.game_list):
                q[action] = score
            else:
                next_state, next_action = self.game_list[i]
                updated_q = self.neur_net.forward_pass(next_state)
                q[action] += self.gamma * updated_q[action]
            self.neur_net.backProp(state, self.neur_net.mkVec(q))
            if i != len(self.game_list):
                action, q = next_action, updated_q

class HumanPlayer:
    def play(self, place_func, board_state, me, _):
        pos = map(int, map(str.strip, input().split(" ")))
        print(pos)
        place_func(*pos)
        return True

class RandomPlayer:
    def play(self, place_func, board_state, me, _):
        comp = False
        pos = None
        x = []
        for i in range(8):
            for j in range(8):
                x.append((i, j))
        random.shuffle(x)
        while not comp and x:
            pos = x.pop()
            comp = place_func(*pos)
        if not comp and not x:
            return False
