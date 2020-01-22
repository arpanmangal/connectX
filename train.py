"""
Class containing the player which interacts with Monte-Carlo to learn and play the game when reqd
"""

import numpy as np
import traceback
from montecarlo import MonteCarlo
from net import NeuralTrainer
import time, os
from joblib import Parallel, delayed
import random
import pickle
import math
from tqdm import tqdm
import gc
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Player:
    """
    Self-play bot, generating thousands of MCTS games, and training the neural network
    """
    def __init__ (self, nrows=6, ncols=7, inarow=4, mcts_sims=200, num_games=10, batch_size=10000, running_batch_file='running_batch.pkl', fnet=None,  load_running_batch=False):
        """
        Initialize the Player class, instantiating the monte-carlo and Fnetwork
        """
        self.nrows = nrows
        self.ncols = ncols
        self.inarow = inarow
        self.mcts_sims = mcts_sims
        self.num_games = num_games # Update the network after generating these many number of games
        self.batch_size = batch_size # Size of the running batch
        self.running_batch_file = running_batch_file # File for loading and saving the running file
        if not load_running_batch:
            # Put an empty batch inside the file
            with open(self.running_batch_file, 'wb') as f:
                pickle.dump([], f)

        # Create the network
        self.fnet = NeuralTrainer(3, input_stack=2, nrows=6, ncols=7, epochs=1, batch_size=256, lr=0.05)
        if fnet is not None:
            # Load the network from the file
            self.fnet.load_model(fnet)

    def _generate_game_batch (self):
        """
        A function independent of the big structures of the Player class
        """

        def play_game(nrows, ncols, inarow, fnet, mcts_sims):
            """
            Generate a single game and batch
            """
            game_batch = []
            try:
                eprint ("Instantiating Sim")
                simulator = MonteCarlo(nrows, ncols, inarow, fnet, mcts_sims) # Create a MCTS simulator
                game_batch = simulator.play_game()
            except:
                tb = traceback.format_exc()
                raise
            # else:
            #     tb = "No error"
            # finally:
            #     eprint(tb)
            return game_batch

        games_batch = Parallel(n_jobs=self.num_games)(delayed(play_game)(self.nrows, self.ncols, self.inarow, self.fnet, self.mcts_sims) for i in range(self.num_games))
        batch = [b for gb in games_batch for b in gb]
        gc.collect()

        return batch
    
    def self_play(self, total_games=10, checkpoint_path=None,  logging=True, log_file=None, game_offset=0):
        """
        Generate games from self-play and update the network
        """
        num_times = int(np.ceil(total_games / self.num_games))
        for g in range(num_times):
            game = g + game_offset
            checkpoint_file = os.path.join(checkpoint_path, 'net%d.model' % game)

            # Generate a game
            print('\n\n\n######################################################################')
            print ('GAME %d' % game)
            print('#######################################################################\n\n')

            start_time = time.time()
            batch = self._generate_game_batch()
            random.shuffle(batch)
            end_time = time.time()
            eprint("GAME # %d | Time Taken: %.3f secs" % ((game + 1) * self.num_games, end_time - start_time))

            # Update the network
            self.update_network(batch, checkpoint_path=checkpoint_file, logging=logging, log_file=log_file)
            gc.collect()

    def _chunks(self, running_batch, chunk_size, num_chunks):
        """Yield successive n-sized chunks from l."""
        for _i in range(num_chunks):
            yield [random.choice(running_batch) for _c in range(chunk_size)] # With replacement

    def update_network(self, batch, checkpoint_path=None, logging=True, log_file=None):
        """
        Update the weights of the network
        """
        start_time = time.time()
        new_batch = self._augment_batch(batch)
        random.shuffle(new_batch)

        # Load and modify the running batch
        with open(self.running_batch_file, 'rb') as f:
            running_batch = pickle.load(f)

        running_batch += new_batch
        running_batch = running_batch[-self.batch_size:]

        if len(running_batch) >= self.batch_size // 4:
            # Start training only after generating quarter full batch
            for chunk in tqdm(self._chunks(running_batch, len(new_batch) // 4, 8) ):
                self.fnet.train(chunk, logging=logging, log_file=log_file)

        # Save the network
        if checkpoint_path is not None:
            self.fnet.save_model(checkpoint_path)

        # Save the running batch
        with open(self.running_batch_file, 'wb') as f:
            pickle.dump(running_batch, f)

        if log_file is not None:
            with open(log_file, 'a') as lf:
                lf.write('\nTrained on running_batch of size %d/%d\n' % ((len(new_batch) // 4) * 8, len(running_batch)))
                lf.write('---------------------------------------------------------------------------------------\n\n')

        eprint ('Network Updated. Time Taken: %d secs' % (time.time() - start_time))
        gc.collect()

    def _augment_batch(self, batch):
        """
        For each example in the batch, augment it with rotations and flips
        """
        new_batch = []
        for (s, pi, r) in batch:
            states = self._transform_state(s)
            policies = self._transform_policy(pi)

            for s_t, pi_t in zip(states, policies):
                new_batch.append((s_t, pi_t, r))

        return new_batch

    def _transform_state(self, stack):
        return [stack, np.flip(stack, 2)]

    def _transform_policy(self, policy):
        return [policy, self.ncols - 1 - policy]

if __name__ == '__main__':
    # Create a player
    player = Player(mcts_sims=500, num_games=20, batch_size=25000, fnet=None, running_batch_file='models/jan22/running_batch.pkl', load_running_batch=False)
    player.self_play(10000, 'models/jan22/', logging=True, log_file='models/jan22/training.txt', game_offset=0)