from random import sample
from random import * 
from torch.nn import MSELoss

from python.constantes import *

import numpy as np

import copy
import numpy as np
import torch

from python.NeuronalNetwork import *
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, action_space, observation_space, cuda):
        self.action_space = action_space
        self.buffer = []
        self.buffer_size = BUFFER_SIZE

        self.alpha = TAU
        self.gamma = GAMMA

        self.i = 0
        self.tab_erreur = []

        self.noise = OUNoise(action_space.shape[0])

        if(False):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.critic = Critic(observation_space.shape[0], action_space.shape[0]).to(device=self.device)
        self.critic_target = copy.deepcopy(self.critic).to(device=self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LR_CRITIC, weight_decay=WEIGHT_DECAY) # smooth gradient descent

        self.actor = Actor(observation_space.shape[0], action_space.shape[0]).to(device=self.device)
        self.actor_target = copy.deepcopy(self.actor).to(device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LR_ACTOR) # smooth gradient descent
        


    def act(self, observation, reward, done):
        action = self.actor(torch.tensor(observation,  dtype=torch.float32, device=self.device)).data.numpy()
        noise = self.noise.sample()
        action += noise
        return torch.tensor(np.clip(action, self.action_space.low[0], self.action_space.high[0], action))
        

    def sample(self, n=BATCH_SIZE):
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
        spl = self.sample()


        tens_ob = torch.tensor([item[0] for item in spl], dtype=torch.float32, device=self.device)
        tens_action = torch.tensor([[item[1]] for item in spl], dtype=torch.long, device=self.device)
        tens_ob_next = torch.tensor([item[2] for item in spl], dtype=torch.float32, device=self.device)
        tens_reward = torch.tensor([item[3] for item in spl], dtype=torch.float32, device=self.device)
        tens_done = torch.tensor([item[4] for item in spl], dtype=torch.float32, device=self.device)

        tens_qvalue = self.critic(tens_ob, tens_action.float()).squeeze()

        tens_next_action = self.actor_target(tens_ob_next)

        tens_next_qvalue = self.critic_target(tens_ob_next, tens_next_action).squeeze()
        
        self.critic_optimizer.zero_grad()
        critic_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done)
        critic_loss.backward()
        self.critic_optimizer.step()

        tens_actions_pred = self.actor(tens_ob)
        actor_loss = -self.critic(tens_ob, tens_actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )




class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # Thanks to Hiu C. for this tip, this really helped get the learning up to the desired levels
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx
        return self.state
            

