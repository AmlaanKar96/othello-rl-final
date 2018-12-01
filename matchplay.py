from game import Game
from players import *

player1 = RandomPlayer()
player = QPlayer(1,1)
player.neur_net.load("Q-Self.weights")
#player1.policy_net.load("best-linear-0.03.weights")

g = Game()
g.new_game(player, player1)
g.game_play(1)
