from random import sample
from random import * 
from torch.nn import MSELoss

import numpy as np

import copy
import numpy as np
import torch

from python.hyperParams import hyperParams as hp

from python.NeuralNetworks import *

class DiscreteAgent(object):
    def __init__(self, action_space, observation_space, cuda, hyperParams=None, actor_to_load=None):

        if(hyperParams == None):
            self.hyperParams = hp
        else:
            self.hyperParams = hyperParams
            self.hyperParams.EPSILON = 0

        self.action_space = action_space    
        self.buffer = []
        self.buffer_size = self.hyperParams.BUFFER_SIZE

        self.alpha = self.hyperParams.ALPHA
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.actor = ActorRNN(observation_space.shape[0], action_space.n)

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None):
            self.actor.load_state_dict(torch.load(actor_to_load))
            self.actor.eval()
        
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

        


    def act(self, observation):
        #return self.action_space.sample()
        tens_action = self.actor(torch.Tensor(observation[0]).unsqueeze(0), torch.Tensor(observation[1]).unsqueeze(0))
        tens_action = tens_action.squeeze()
        rand = random()
        if(rand > self.epsilon):
            _, indices = tens_action.max(0)
            return indices.item()
        return randint(0, tens_action.size()[0]-1)

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return sample(self.buffer, len(self.buffer))
        else:
            return sample(self.buffer, self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self, n_iter):
        #n_iter only used in continuousAgent
        loss = MSELoss()
        '''if(self.epsilon > self.hyperParams.MIN_EPSILON):
            self.epsilon *= self.hyperParams.EPSILON_DECAY
        else:
            self.epsilon = 0'''

        self.epsilon -= self.hyperParams.EPSILON_DECAY

        spl = self.sample()

        tens_path = torch.Tensor([item[0][0] for item in spl])
        tens_state = torch.Tensor([item[0][1] for item in spl])
        tens_action = torch.LongTensor([item[1] for item in spl])
        tens_path_next = torch.Tensor([item[2][0] for item in spl])
        tens_state_next = torch.Tensor([item[2][1] for item in spl])
        tens_reward = torch.Tensor([item[3] for item in spl])
        tens_done = torch.Tensor([item[4] for item in spl])

        tens_qvalue = self.actor(tens_path, tens_state)
        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag()


        tens_next_qvalue = self.actor_target(tens_path_next, tens_state_next)
        tens_next_qvalue = torch.max(tens_next_qvalue, 1)[0]

        self.optimizer.zero_grad()
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done)
        tens_loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )