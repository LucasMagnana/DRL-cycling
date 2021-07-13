import torch
from torch import nn
from torch.autograd import Variable
from python.constantes import *



class Actor(nn.Module):

    def __init__(self, size_ob, size_action):
        super(Actor, self).__init__()
        self.inp = nn.Linear(size_ob, ACT_IN)
        self.int = nn.Linear(ACT_IN, ACT_INTER)
        self.out = nn.Linear(ACT_INTER, size_action)

    def forward(self, ob):
        out = nn.functional.relu(self.inp(ob))
        out = nn.functional.relu(self.int(out))
        return torch.tanh(self.out(out))


class Critic(nn.Module):

    def __init__(self, size_ob, size_action):
        super(Critic, self).__init__()
        self.inp1 = nn.Linear(size_ob, CRIT_IN)
        self.inp2 = nn.Linear(size_action, CRIT_IN)
        self.int = nn.Linear(CRIT_IN*2, CRIT_INTER)
        self.out = nn.Linear(CRIT_INTER, 1)

    def forward(self, ob, action):
        out1 = nn.functional.relu(self.inp1(ob))
        out2 = nn.functional.relu(self.inp2(action))
        out = nn.functional.relu(self.int(torch.cat((out1, out2), 1)))
        return self.out(out)

