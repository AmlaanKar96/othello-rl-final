import sys
from collections import defaultdict
import numpy as np
from colorama import init, Fore, Back, Style
init(autoreset=True)


class Board(object):
    BLACK = 1
    WHITE = -1

    def __init__(self):
        self.board = np.zeros((8,8), int)
        self.board[3][3] = Board.BLACK
        self.board[4][4] = Board.BLACK
        self.board[4][3] = Board.WHITE
        self.board[3][4] = Board.WHITE

        self.remaining_squares = 8*8 - 4
        self.score = {Board.BLACK: 2, Board.WHITE: 2}

    def getScore(self):
        return self.score

    def getState(self):
        return self.board

    def isOnBoard(self, x, y):
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    def updateBoard(self, tile, row, col):
        result = self.isValidMove(tile, row, col)
        if result:
            self.board[row][col] = tile
            for row in result:
                self.board[row[0]][row[1]] = tile
            self.score[tile] += len(result) + 1
            self.score[(((tile + 1) // 2 + 1) % 2) * 2 - 1] -= len(result)
            self.remaining_squares -= 1
            return True
        else:
            return False

    def isValidMove(self, tile, xstart, ystart):
        if not self.isOnBoard(xstart, ystart) or self.board[xstart][ystart] != 0:
            return False
        self.board[xstart][ystart] = tile

        otherTile = tile * -1

        tiles_to_flip = []
        for xdirection, ydirection in ((0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)):
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            if self.isOnBoard(x, y) and self.board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not self.isOnBoard(x, y):
                    continue
                while self.board[x][y] == otherTile:
                    x += xdirection
                    y += ydirection
                    if not self.isOnBoard(x, y):
                        break
                if not self.isOnBoard(x, y):
                    continue
                if self.board[x][y] == tile:
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tiles_to_flip.append([x, y])
        self.board[xstart][ystart] = 0
        return tiles_to_flip

    def printBoard(self):
        def getItem(item):
            if item == Board.BLACK :
                return Fore.WHITE + "|" + Fore.BLACK + "O"
            elif item == Board.WHITE :
                return Fore.WHITE + "|" + Fore.WHITE + "O"
            else:
                return Fore.WHITE + "| "

        def getRow(row):
            return "".join(map(getItem,row))

        print("\t" +              "      BOARD      ")
        print("\t" + Fore.WHITE + " |0|1|2|3|4|5|6|7")
        for i in range(8):
            print("\t" +  Fore.WHITE + "{}{}".format(i,
                getRow(self.board[i])))
            sys.stdout.write(Style.RESET_ALL)
