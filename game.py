import numpy as np
import board

class Game:
    def __init__(self):
        self.board = board.Board()
        self.players = []

    def new_game(self, player, player1, log_move_history = True, log_move_history1 = True):
        self.players.append((player, log_move_history))
        self.players.append((player1, log_move_history1))

    def getScore(self):
        return self.board.getScore()

    def game_play(self, show_board = 0):
        num = 0
        while num < 2:
            num = 0
            for i, player in enumerate(self.players):
                func = i * 2 - 1
                did_move = player[0].play(lambda r,c: self.board.updateBoard(func,r,c), self.board.getState(), func, player[1])

                if show_board == 1:
                    self.board.printBoard()

                if did_move is not True:
                    num += 1

