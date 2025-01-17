"""
Module containing the code for MonteCarlo class

Generates ONE complete game using the current FNetwork
Runs simulations and return a batch to update the neural network
"""

import numpy as np
from simulator import ConnectX
from net import NeuralTrainer
import traceback
import time

class MonteCarlo:
    def __init__ (self, nrows, ncols, inarow, net : NeuralTrainer, max_sims : int = 200, tau_thres : int = 6, mode : str = 'selfplay'):
        # Initialize the MonteCarlo class
        self.nrows = nrows
        self.ncols = ncols
        self.inarow = inarow
        self.net = net
        self.max_sims = max_sims
        self.mode = mode

        assert self.mode in ['selfplay', 'opp']
        if self.mode == 'opp':
            raise NotImplementedError

        np.set_printoptions(precision=3)

        # Hyperparameters
        self.cpuct = 1.0
        self.tau_thres = tau_thres

        # Set of (s, pi, r) tuples
        # s here is the complete 2*nrows*ncols state
        self.batch = []
        
        # Tracking the values
        self.Qsa = dict() # Stores Q values for s,a pairs
        self.Nsa = dict() # Stores the count for s,a pairs
        self.Ns = dict() # Count of number of times s is encountered

        self.Ps = dict() # Stores initial policy returned by the Fnet
        self.Ms = dict() # Stores list of valid moves
        self.Ts = dict() # Terminal states

    def play_game (self):
        """
        Play one full game, simulating on each move
        Returns the batch of (s, pi, r) tuples, for updating the fnet
        """
        # Initial state -- an instance of GoEnv
        self.state = ConnectX(nrows=self.nrows, ncols=self.ncols, inarow=self.inarow)
        self.state.reset()
        root_state = True # Whether this is the first state

        while not self.state.is_over():
            print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print ('Ns: %d | Qsa: %d | Ms: %d' % (len(self.Ns), len(self.Qsa), len(self.Ms)))
            print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            # Perform a simulation on the COPY of current state
            for _sim in range(self.max_sims):
                start_state = self.state.create_copy()
                self.run_simulator(start_state)

            # Compute the policy from the root node and add to the batch
            # Add dummy reward to the batch for now, update at end of game
            policy = self._compute_pi(self.state)
            self.batch.append((self.state.get_stack(), policy, self.state.get_player_turn()))

            # Update state and delete not-needed tree
            print ('Player: ', self.state.get_player_turn())
            self.play_move(policy[:], root_state=root_state)
            self.state.print_board()
            root_state = False
            print ('Over?', self.state.is_over())
            print ('-----------------------------------------------------------------')

        # Update the reward and return the batch
        winner = self.state.get_winner()
        print('######################')
        self.state.print_board()
        print ("And the winner is .... %s !" % ('One' if winner == 1 else ('Two' if winner == -1 else 'Draw')))
        print ('=========================================================================================')

        for idx, (s, pi, pturn) in enumerate(self.batch):
            r = winner * pturn
            self.batch[idx] = (s, np.argmax(pi), r)

        return self.batch

    def _compute_pi(self, state):
        """
        Compute the policy proportional N(s,a)
        """
        # Get the board representation of state
        s = state.hash_state()
        valid_moves = state.get_legal_moves()
        assert valid_moves.shape[0] == self.ncols

        counts = np.array([self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.ncols)])
        print (counts, valid_moves)
        counts *= valid_moves # Masking with valid moves

        if np.sum(counts) == 0:
            print("All counts had to be masked :( !!")
            counts = valid_moves

        if self.tau_thres > 0:
            # Take action with proportional probabilities
            self.tau_thres -= 1
            return counts / float(np.sum(counts))
        else:
            max_policy = np.zeros(self.ncols)
            max_policy[np.argmax(counts)] = 1.0
            return max_policy

    def play_move(self, policy, root_state):
        """
        Choose an action according to the policy from the current state
        Execute and go to the next state
        """
        if root_state:
            policy = np.zeros(self.ncols)
            policy[np.random.randint(self.ncols)] = 1.0

        a = np.random.choice(np.arange(0, self.ncols), p=policy)
        self.state.step(a)
        print (policy)
        print("Played %s" % a)

    def run_simulator(self, state):
        """
        Run one iteration of the MCTS simulation from the 'root': state
        Fig. 2 of paper
        a. Use the UCT to expand
        b. Once we hit the leaf node, use the FNet to compute values
        c. Update the path to the root with the value
        """
        # Get the board representation of state
        s = state.hash_state()
        stack = state.get_stack()

        if s not in self.Ts:
            if state.is_over():
                self.Ts[s] = state.get_player_turn() * state.get_winner()
                assert self.Ts[s] != 1 # I can't be the winner if it's my turn next
            else:
                self.Ts[s] = 'Not Over'

        if self.Ts[s] == 1 or self.Ts[s] == -1 or self.Ts[s] == 0:
            # This is a terminal state
            return -self.Ts[s]

        if s not in self.Ps:
            # Leaf node
            p, v = self.net.predict(stack)
            valid_moves = state.get_legal_moves()
            p = p * valid_moves # masking invalid moves
            sum_p = np.sum(p)
            if sum_p > 0:
                p /= sum_p
            else:
                print ('All valid moves had to be masked!!')
                print (p, valid_moves)
                p = valid_moves / np.sum(valid_moves)
            
            self.Ms[s] = valid_moves
            self.Ps[s] = p
            self.Ns[s] = 0

            return -v

        # Pick the action with highest confidance bound
        s = state.hash_state()
        valid_moves = self.Ms[s]
        best = -float('inf')
        best_action = -1

        def get_Q_plus_U(s, a):
            if (s,a) in self.Qsa:
                return self.Qsa[(s,a)] + \
                        self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
            else:
                # Taking Q(s,a) = 0 here
                return self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)

        for a in range(self.ncols):
            if valid_moves[a]:
                value = get_Q_plus_U(s, a)
                if value > best:
                    best = value
                    best_action = a

        assert 0 <= best_action <= self.ncols - 1

        # Play the best action
        a = best_action
        state.step(a)

        # Recursively call simulator on next state
        v = self.run_simulator(state)

        # Update Qsa
        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v