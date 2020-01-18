"""
minimax player
"""

import numpy as np
from kaggle_environments import evaluate, make, utils


def make_board(board, nrows, ncols):
    assert len(board) == nrows * ncols
    board = np.array(board).reshape(nrows, ncols)
    board[board == 2] = -1
    return board

def utility(board, inarow):
    """
    get utility w.r.t. player 1
    """
    nrows, ncols = board.shape

    def _utility(board):
        utility = 0.0
        def val(s):
            if s == inarow:
                return 1
            # elif s == inarow - 1:
            #     return 1 / 4
            # elif s == inarow - 2:
            #     return 1 / 12
            else:
                return 0

        for r in range(nrows):
            for c in range(ncols - inarow + 1):
                s = np.sum(board[r, c:c+inarow])
                utility += val(s)

        for c in range(ncols):
            for r in range(nrows - inarow + 1):
                s = np.sum(board[r:r+inarow, c])
                utility += val(s)

        # bottom right diagonals
        for rs in range(nrows - inarow + 1):
            for rc in range(ncols - inarow + 1):
                s = 0
                for i in range(inarow):
                    s += board[rs + i][rc + i]
                utility += val(s)

        # bottom left diagonals
        for rs in range(nrows - inarow + 1):
            for rc in range(ncols - 1, inarow - 2, -1):
                s = 0
                for i in range(inarow):
                    s += board[rs + i][rc - i]
                utility += val(s)

        return utility

    return _utility(board) + _utility(-1 * board)

def is_over(board, inarow):
    u = utility(board, inarow)
    if u == 1 or u == -1:
        return True
    else:
        return False

def valid_move(board, col):
    return board[0][col] == 0

def play_move(board, toplay, col):
    assert valid_move(board, col)

    if board[-1][col] == 0:
        board[-1][col] = toplay
        return
    r = 0
    while board[r][col] == 0: r+=1
    board[r-1][col] = toplay

def unplay(board, col):
    r = 0
    while board[r][col] == 0: r += 1
    board[r][col] = 0

def oneply(board, toplay, inarow):
    nrows, ncols = board.shape
    utils = []
    for col in range(ncols):
        if valid_move(board, col):
            play_move(board, toplay, col)
            u = toplay * utility(board, inarow)
            unplay(board, col)
        else:
            u = -100
        utils.append(u)

    move = int(np.argmax(utils))
    return move, utils

def minimax(board, toplay, inarow, depth):
    # print (board, type(board), board.shape)
    nrows, ncols = board.shape
    # print ('hey')

    if depth == 1:
        return oneply(board, toplay, inarow)

    utils = []
    for col in range(ncols):
        if valid_move(board, col):
            play_move(board, toplay, col)
            if not is_over(board, inarow):
                move, all_utils = minimax(board, -toplay, inarow, depth - 1)
                u = -1 * all_utils[move]
            else:
                u = toplay * utility(board, inarow)
            # u = toplay * utility(board, inarow)
            unplay(board, col)
        else:
            u = -100
        utils.append(u)

    move = int(np.argmax(utils))
    return move, utils

def agent(observation, configuration):
    board = make_board(observation['board'], configuration['rows'], configuration['columns'])
    inarow = configuration['inarow']
    toplay = 1 if observation['mark'] == 1 else -1

    # move, _ = oneply(board, toplay, inarow)
    move, utils = minimax(board, toplay, inarow, 2)
    print (move, utils)
    return move
    