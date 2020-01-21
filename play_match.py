"""
Play match between two neural networks
"""

from net import NeuralTrainer
from simulator import ConnectX
import numpy as np
from tqdm import tqdm

def play_matches(net1, net2, num_matches=4):
    fnet1 = NeuralTrainer(4, input_stack=2, nrows=6, ncols=7)
    fnet1.load_model(net1)

    fnet2 = NeuralTrainer(4, input_stack=2, nrows=6, ncols=7)
    fnet2.load_model(net2)

    score1 = 0; score2 = 0
    for game in tqdm(range(num_matches)):
        if game % 2 == 0:
            players = {'1': 1, '-1': 2}
        else:
            players = {'1': 2, '-1': 1}

        env = ConnectX()
        while not env.is_over():
            player_turn = env.get_player_turn()
            if players[str(player_turn)] == 1:
                # print ('using fnet1')
                moves, _ = fnet1.predict(env.get_stack())
            else:
                # print ('using fnet2')
                moves, _ = fnet2.predict(env.get_stack())

            valid_moves = env.get_legal_moves()
            moves *= valid_moves
            if (np.sum(moves) == 0):
                moves = valid_moves

            move = np.argmax(moves)
            env.step(move)

        winner = env.get_winner()
        if players[str(winner)] == 1:
            score1 += 1
        else:
            score2 += 1

    print (score1, score2)


if __name__ == '__main__':
    play_matches('models/jan19/net10.model', 'models/jan19/net30.model')
