import numpy as np
from game import Game
from players import QPlayer
from players import SarsaPlayer
from players import RandomPlayer
import matplotlib.pyplot as plt

player = QPlayer(0.79, 0.05)
player1 = QPlayer(0.79, 0.05)
old_score = 0
rp = RandomPlayer()
no_of_games = 100
epochs = 2000
player_wins = []
#player_wins1 = []
for e in range(epochs):
    print("Epoch: %d"%e)
    player.wins = 0
    #player1.wins = 0
    player.explore_rate = np.exp(-0.017*e) / 0.11 + 0.1
    g = Game()
    g.new_game(player, player1)
    g.game_play()
    result = g.getScore()
    res = list(result.items())
    res.sort()
    x = res[0][1]
    player_score =  x / 32 - 1
    if e > 0:
        player1.weight_update(old_score)
    player.weight_update(player_score)
    old_score = player_score
    #if e % no_of_games == 0 and e != 0:
    for i in range(10):
        player.game_list = []
        g = Game()
        g.new_game(player, rp, True, False)
        g.game_play()
        result = g.getScore()
        res = list(result.items())
        res.sort()
        x = res[0][1]
        player_score =  (x / 64 - 0.5)*2
        player.wins += player_score > 0
        player_wins.append(player.wins / no_of_games)
    #player.wins += player_score > 0
    #player_wins.append(player.wins / no_of_games)
    #player1.wins += player_score > 0
    #player_wins1.append(player1.wins / no_of_games)

player.policy_net.save("Learned weights: Q")

plt.plot(player_wins)
plt.draw()
plt.show()
#plt.plot(player_wins)
#plt.plot(player_wins1)
#plt.draw()
#plt.show()
