from random import sample
from random import * 
from torch.nn import MSELoss

import numpy as np

import copy
import numpy as np
import torch

from python.NeuronalNetwork import NN
import matplotlib.pyplot as plt

class AgentStick(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, cuda):
        self.action_space = action_space    
        self.buffer = []
        self.buffer_size = 100000

        self.alpha = 0.05
        self.epsilon = 1.0
        self.gamma = 0.9

        self.i = 0
        self.tab_erreur = []

        if(False):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


        self.neur = NN().to(device=self.device)
        self.neur_target = copy.deepcopy(self.neur).to(device=self.device)

        self.optimizer = torch.optim.Adam(self.neur.parameters(), 0.001) # smooth gradient descent
        


    def act(self, observation, reward, done):
        #return self.action_space.sample()
        tens_action = self.neur(torch.tensor(observation,  dtype=torch.float32, device=self.device))
        rand = random()
        if(rand > self.epsilon):
            _, indices = tens_action.max(0)
            return indices.item()
        return randint(0, tens_action.size()[0]-1)

    def sample(self, n=128):
        if(n > len(self.buffer)):
            n = len(self.buffer)
        return sample(self.buffer, n)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self):
        self.i += 1
        loss = MSELoss()
        if(self.epsilon > 0.1):
            self.epsilon *= 0.99
        spl = self.sample()
        '''for screen in spl :
            tensor_qvalues = self.neur(torch.Tensor(screen[0]))
            qvalue = tensor_qvalues[screen[1]]
            reward = screen[3]
            next_qvalues = self.neur_target(torch.Tensor(screen[2]))
            max_next_qvalues = torch.max(next_qvalues)
            self.optimizer.zero_grad()
            if(screen[4]):
                loss_tmp = loss(qvalue, reward+(self.gamma*max_next_qvalues))
            else :
                loss_tmp = loss(qvalue, torch.Tensor(np.array(reward)))
            loss_tmp.backward()
            self.optimizer.step()'''

        tens_ob = torch.tensor([item[0] for item in spl], dtype=torch.float32, device=self.device)
        tens_action = torch.tensor([item[1] for item in spl], dtype=torch.long, device=self.device)
        tens_ob_next = torch.tensor([item[2] for item in spl], dtype=torch.float32, device=self.device)
        tens_reward = torch.tensor([item[3] for item in spl], dtype=torch.float32, device=self.device)
        tens_done = torch.tensor([item[4] for item in spl], dtype=torch.float32, device=self.device)

        tens_qvalue = self.neur(tens_ob)
        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag()


        tens_next_qvalue = self.neur_target(tens_ob_next)
        tens_next_qvalue = torch.max(tens_next_qvalue, 1)[0]

        self.optimizer.zero_grad()
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done)
        tens_loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.neur_target.parameters(), self.neur.parameters()):
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )
            

