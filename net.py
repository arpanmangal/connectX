import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, padding=1):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=padding)
        self.bn = batchnorm
        self.b1 = nn.BatchNorm2d(out_filters)
        
    def forward(self, x):
        x = self.c1(x)
        if self.bn: 
            x = self.b1(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, batchnorm=True, activ=F.relu, padding=1):
        '''
        filters: number of input and output filters for convolution layer
        if modifying kernel_size adjust the padding too
        '''
        super(ResidualBlock, self).__init__()
        self.activ = activ
        self.conv1 = ConvBlock(in_filters=filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.conv2 = ConvBlock(in_filters=filters, out_filters=filters, 
                               kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
    
    def forward(self, x):
        x_mod = self.activ(self.conv1(x))
        x_mod = self.conv2(x_mod)
        x_mod = self.activ(x_mod + x) # Skip-connection
        return x_mod


class PolicyHead(nn.Module):
    def __init__(self, filters=2, kernel_size=1, batchnorm=True, activ=F.relu, nrows=6, ncols=7, res_filters=256, padding=0):
        super(PolicyHead, self).__init__()
        self.conv = ConvBlock(in_filters=res_filters, out_filters=filters, 
                              kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.activ = activ
        self.fc_infeatures = filters * nrows * ncols
        self.fc = nn.Linear(in_features=self.fc_infeatures, out_features=ncols)
    
    def forward(self, x):
        x = self.activ(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, filters=1, kernel_size=1, batchnorm=True, activ=F.relu, nrows=6, ncols=7, res_filters=256, padding=0):
        super(ValueHead, self).__init__()
        self.conv = ConvBlock(in_filters=res_filters, out_filters=filters, 
                              kernel_size=kernel_size, batchnorm=batchnorm,  padding=padding)
        self.activ = activ
        self.fc_infeatures = filters * nrows * ncols
        self.fc1 = nn.Linear(in_features=self.fc_infeatures, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)
    
    def forward(self,x):
        x = self.activ(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1)
        return torch.tanh(x)


class AlphaNeural(nn.Module):
    def __init__(self, res_blocks, input_stack, nrows, ncols):
        super(AlphaNeural, self).__init__()
        self.res_blocks = res_blocks
        self.nrows = nrows
        self.ncols = ncols
        self.input_stack = input_stack
        self.res_filters = 48

        self.conv1 = ConvBlock(in_filters=self.input_stack, out_filters=self.res_filters)
        for i in range(self.res_blocks):
            self.add_module("ResBlock%d" % i, ResidualBlock(filters=self.res_filters))

        self.policy_head = PolicyHead(res_filters=self.res_filters)
        self.value_head = ValueHead(res_filters=self.res_filters)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        for i in range(self.res_blocks):
            x = self._modules["ResBlock%d" % i](x)

        action = self.policy_head(x)
        value = self.value_head(x)

        return action, value

class NeuralTrainer():
    def __init__(self, res_blocks, input_stack=2, nrows=6, ncols=7, lr=1e-3, epochs=10, batch_size=64):
        self.nrows = nrows
        self.ncols = ncols
        self.net = AlphaNeural(res_blocks=res_blocks, input_stack=input_stack, nrows=nrows, ncols=ncols)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()

    def train(self, examples, logging=True, log_file=None):
        '''examples is list containing (board, policy, value)'''

        self.net.train()

        for i, e in enumerate(examples):
            try:
                assert 0 <= e[1] <= self.ncols - 1
                assert -1 <= e[2] <= 1
            except:
                print (e)
                print (i)
                raise

        states = torch.FloatTensor([x[0] for x in examples])
        policies = torch.LongTensor([x[1] for x in examples])
        values = torch.FloatTensor([x[2] for x in examples])

        train_dataset = TensorDataset(states, policies, values)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.net.parameters(), weight_decay=1e-4)
        val_criterion = nn.MSELoss(reduction="mean")
        pi_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            policy_loss = 0.0
            value_loss = 0.0
            for s, pi, v in train_loader:
                if self.cuda_flag:
                    s, pi, v = s.cuda(), pi.cuda(), v.contiguous().cuda()

                # Forward pass
                opt.zero_grad()
                pred_pi, pred_v = self.net(s)

                pi_loss = pi_criterion(pred_pi, pi)
                val_loss = val_criterion(pred_v, v)
                total_loss = pi_loss + val_loss

                policy_loss += pi_loss.item()
                value_loss += val_loss.item()

                # Backprop
                total_loss.backward()
                opt.step()

            policy_loss /= len(train_loader)
            value_loss /= len(train_loader)

            if logging:
                print("Epoch:{} Policy Loss:{} Value Loss:{}".format(epoch, policy_loss, value_loss))
        
            if log_file is not None:
                with open(log_file, 'a') as f:
                    timestamp = str(datetime.datetime.now()).split('.')[0]
                    f.write("{} | Epoch:{} Policy Loss:{} Value Loss:{}".format(timestamp, epoch, policy_loss, value_loss))

        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write('\n') # An extra space after each training

    def predict(self, board):
        """ predict value and policy """
        board = torch.FloatTensor(board)
        if self.cuda_flag:
            board = board.contiguous().cuda()

        single = False
        if board.dim() == 3:
            # It's a single input
            single = True
            board = board.unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board)
            pi = F.softmax(pi, dim=1)

        if single:
            return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        return pi.data.cpu().numpy(), v.data.cpu().numpy()

    def save_model(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path))

    def load_model_dict(self, d):
        self.net.load_state_dict(d)

