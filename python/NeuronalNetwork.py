import torch
from torch import nn
from torch.autograd import Variable
from python.constantes import *



class Actor(nn.Module):

    def __init__(self, size_ob):
        super(Actor, self).__init__()
        self.inp = nn.Linear(size_ob, ACT_IN)
        self.int = nn.Linear(ACT_IN, ACT_INTER)
        self.out = nn.Linear(ACT_INTER, 1)

    def forward(self, ob):
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        return self.out(out)


class Critic(nn.Module):

    def __init__(self, size_ob, size_action):
        super(Critic, self).__init__()
        self.inp = nn.Linear(size_ob, CRIT_IN)
        self.int = nn.Linear(CRIT_IN+size_action, CRIT_INTER)
        self.out = nn.Linear(CRIT_INTER, 1)

    def forward(self, ob, action):
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(torch.cat((out, action), 1)))
        return self.out(out)

