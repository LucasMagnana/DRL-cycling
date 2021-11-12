import torch
from torch import nn
from torch.autograd import Variable

import python.hyperParams as HP



class Actor(nn.Module):

    def __init__(self, size_ob, size_action, hp_saved=None, max_action=1, tanh=False): #for saved hyperparameters
        super(Actor, self).__init__()
        if(hp_saved == None):
            hyperParams=HP.hyperParams
        else:
            hyperParams=hp_saved
        self.inp = nn.Linear(size_ob, hyperParams.ACT_IN)
        self.int = nn.Linear(hyperParams.ACT_IN, hyperParams.ACT_INTER)
        self.out = nn.Linear(hyperParams.ACT_INTER, size_action)
        self.max_action = max_action
        self.tanh = tanh

    def forward(self, ob):
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        if(self.tanh):
            return torch.tanh(self.out(out)*self.max_action)
        else:
            return self.out(out)*self.max_action



class Critic(nn.Module):

    def __init__(self, size_ob, size_action):
        super(Critic, self).__init__()
        self.inp = nn.Linear(size_ob+size_action, hyperParams.CRIT_IN)
        self.int = nn.Linear(hyperParams.CRIT_IN, hyperParams.CRIT_INTER)
        self.out = nn.Linear(hyperParams.CRIT_INTER, 1)

    def forward(self, ob, action):
        out = nn.functional.relu(self.inp(torch.cat((ob, action), dim=1)))
        out = nn.functional.relu(self.int(out))
        return self.out(out)

