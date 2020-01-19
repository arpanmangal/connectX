import numpy as np


class ConnectX:
    def __init__(self, nrows=6, ncols=7, inarow=4):
        self.nrows = nrows
        self.ncols = ncols
        self.inarow = inarow

        self.board = np.zeros((nrows, ncols), dtype=int)
        self.player_turn = 1 # Two players 1 and 2

    def reset(self):
        # With probability 1/2 play as 1 ow leave
        if np.random.random() > 0.5:
            move = np.random.randint(0, self.ncols)
            self.step(move)

    def player_turn(self):
        return self.player_turn

    def step(self, move):
        assert self.board[0][move] == 0
        if self.board[self.nrows - 1][move] == 0:
            self.board[self.nrows - 1][move] = self.player_turn
        else:
            r = 0
            while self.board[r][move] == 0: r += 1
            self.board[r - 1][move] = self.player_turn

        self.player_turn *= -1

    def valid_moves(self):
        valid_moves = np.zeros(self.ncols, dtype=int)
        valid_moves = (self.board[0] == 0).astype(int)
        return valid_moves

    def is_over(self):
        """
        Returns True if the game has ended
        """
        for r in range(self.nrows):
            for c in range(self.ncols):
                s = np.sum(self.board[r, c:c+self.inarow])
                if s == self.inarow or s == -self.inarow:
                    return True


        for c in range(self.ncols):
            for r in range(self.nrows):
                s = np.sum(self.board[r:r+self.inarow, c])
                if s == self.inarow or s == -self.inarow:
                    return True

        for r in range(self.nrows - self.inarow + 1):
            for c in range(self.ncols - self.inarow + 1):
                s = 0
                for i in range(self.inarow):
                    s += self.board[r+i, c+i]
                if s == self.inarow or s == -self.inarow:
                    return True
        for r in range(self.nrows - self.inarow + 1):
            for c in range(self.ncols - 1, self.inarow - 2, -1):
                s = 0
                for i in range(self.inarow):
                    s += self.board[r+i, c-i]
                if s == self.inarow or s == -self.inarow:
                    return True
        return False

    def create_copy(self):
        """
        Create copy
        """
        copy_sim = ConnectX(self.nrows, self.ncols, self.inarow)
        copy_sim.board[:][:] = self.board[:][:]
        copy_sim.player_turn = self.player_turn
        return copy_sim

    def print_board(self):
        print (self.board, self.player_turn)

    def get_stack(self):
        stack = np.zeros((2, self.nrows, self.ncols), dtype=int)
        stack[0] = self.board == self.player_turn
        stack[1] = self.board == -self.player_turn
        return stack

    def hash_state(self):
        shash = ""
        for c in self.board.reshape(-1).tolist():
            shash += "%d " % c
        return shash

    
    