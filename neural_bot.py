"""
Script for neural bot as well as creating a neural bot submission file
"""
from net import NeuralTrainer
import torch
import numpy as np
import json
import pickle
import zlib
import base64 as b64
import numpy as np

def serializeAndCompress(value, verbose=True):
    serializedValue = pickle.dumps(value)
    if verbose:
        print('Length of serialized object:', len(serializedValue))
    c_data =  zlib.compress(serializedValue, 9)
    if verbose:
        print('Length of compressed and serialized object:', len(c_data))
    return b64.b64encode(c_data)


def get_legal_moves(board, ncols=7):
    valid_moves = np.zeros(ncols, dtype=int)
    valid_moves = (board[0] == 0).astype(int)
    return valid_moves

def neural_agent(observation, configuration):
    assert configuration['rows'] == 6 and configuration['columns'] == 7
    agent = NeuralTrainer(4, 2)
    agent.load_model('../models/jan19/net20.model')

    board = np.array(observation['board'], dtype=int).reshape((6, 7))
    board[board == 2] =  -1
    if observation['mark'] == 1:
        to_play = 1
    elif observation['mark'] == 2:
        to_play = -1
    else:
        raise ValueError

    stack = np.zeros((2, 6, 7), dtype=int)
    stack[0] = (board == to_play).astype(int)
    stack[1] = (board == -to_play).astype(int)

    pi, _ = agent.predict(stack)
    valid_moves = get_legal_moves(board)

    moves = pi * valid_moves
    return int(np.argmax(moves))


def create_submission_file(net):
    model = torch.load(net)
    json_model = dict()
    for key, weights in model.items():
        json_model[key] = weights.tolist()

    # json_model = json.dumps(json_model, indent=2)
    json_model = serializeAndCompress(json_model)

    submission = ""
    submission += """
import json
import pickle
import zlib
import base64 as b64

def decompressAndDeserialize(compresseData):
    d_data_byte = b64.b64decode(compresseData)
    data_byte = zlib.decompress(d_data_byte)
    value = pickle.loads(data_byte)
    return value
"""

    with open('net.py', 'r') as file:
        submission += file.read()
    submission += "\n\n"

    submission += """
def load_model():
    '''
    Loads the model
    '''
    model = decompressAndDeserialize({})

    model_dict = dict()
    for d, w in model.items():
        if torch.cuda.is_available():
            model_dict[d] = torch.tensor(w).cuda()
        else:
            model_dict[d] = torch.tensor(w)

    return model_dict

""".format(json_model)

    submission += """
def get_legal_moves(board, ncols=7):
    valid_moves = np.zeros(ncols, dtype=int)
    valid_moves = (board[0] == 0).astype(int)
    return valid_moves


def agent(observation, configuration):
    assert configuration['rows'] == 6 and configuration['columns'] == 7
    agent = NeuralTrainer(4, 2)
    agent.load_model_dict(load_model())

    board = np.array(observation['board'], dtype=int).reshape((6, 7))
    board[board == 2] =  -1
    if observation['mark'] == 1:
        to_play = 1
    elif observation['mark'] == 2:
        to_play = -1
    else:
        raise ValueError

    stack = np.zeros((2, 6, 7), dtype=int)
    stack[0] = (board == to_play).astype(int)
    stack[1] = (board == -to_play).astype(int)

    pi, _ = agent.predict(stack)
    valid_moves = get_legal_moves(board)

    moves = pi * valid_moves
    return int(np.argmax(moves))
"""

    with open('submission.py', 'w') as f:
        f.write(submission)


if __name__ == '__main__':
    create_submission_file('models/jan19/net20.model')