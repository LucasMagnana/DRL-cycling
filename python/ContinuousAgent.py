from random import sample
from random import * 
import torch.nn.functional as F

from python.hyperParams import hyperParams

import numpy as np

import copy
import numpy as np
import torch
import torch.nn as nn

from python.NeuralNetworks import *
import matplotlib.pyplot as plt


class ContinuousAgent(object):
    def __init__(self, action_space, observation_space, cuda):

        self.buffer_size = hyperParams.BUFFER_SIZE
        self.alpha = hyperParams.TAU
        self.gamma = hyperParams.GAMMA
        self.exploration_noise = hyperParams.EXPLORATION_NOISE
        self.policy_noise = hyperParams.POLICY_NOISE

        self.action_space = action_space
        self.buffer = []

        self.tab_erreur = []

        self.noise = OUNoise(action_space.shape[0])

        if(False):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.critic_1 = Critic(observation_space.shape[0], action_space.shape[0]).to(device=self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1).to(device=self.device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), hyperParams.LR_CRITIC, weight_decay=hyperParams.WEIGHT_DECAY)
        
        self.critic_2 = Critic(observation_space.shape[0], action_space.shape[0]).to(device=self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2).to(device=self.device)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), hyperParams.LR_CRITIC, weight_decay=hyperParams.WEIGHT_DECAY)

        self.actor = Actor(observation_space.shape[0], action_space.shape[0], action_space.high[0], tanh=True, hp_saved=hyperParams).to(device=self.device)
        self.actor_target = copy.deepcopy(self.actor).to(device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), hyperParams.LR_ACTOR)
        


    def act(self, observation, reward, done):
        action = self.actor(torch.tensor(observation,  dtype=torch.float32, device=self.device)).data.numpy()
        action += np.random.normal(0, self.exploration_noise, size=self.action_space.shape[0])
        action = action.clip(self.action_space.low, self.action_space.high)
        return torch.tensor(action).numpy()
        

    def sample(self, n=hyperParams.BATCH_SIZE):
        if(n > len(self.buffer)):
            n = len(self.buffer)
        return sample(self.buffer, n)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self, n_iter):
        
        for i in range(n_iter):
            
            spl = self.sample()

            tens_ob = torch.tensor([item[0] for item in spl], dtype=torch.float32, device=self.device)
            tens_action = torch.tensor([item[1] for item in spl], dtype=torch.long, device=self.device)
            tens_ob_next = torch.tensor([item[2] for item in spl], dtype=torch.float32, device=self.device)
            tens_reward = torch.tensor([item[3] for item in spl], dtype=torch.float32, device=self.device)
            tens_done = torch.tensor([item[4] for item in spl], dtype=torch.float32, device=self.device)
            
            tens_noise = torch.empty(tens_action.shape)
            tens_noise = nn.init.normal_(tens_noise, mean=0, std=self.policy_noise)
            tens_noise = tens_noise.clamp(-hyperParams.NOISE_CLIP, hyperParams.NOISE_CLIP)
            tens_next_action = (self.actor_target(tens_ob_next) + tens_noise)
            tens_next_action = tens_next_action.clamp(-self.action_space.high[0], self.action_space.high[0])

            tens_target_qvalue_1 = self.critic_1_target(tens_ob_next, tens_next_action.float()).squeeze()
            tens_target_qvalue_2 = self.critic_2_target(tens_ob_next, tens_next_action.float()).squeeze()       
            tens_target_qvalue = torch.min(tens_target_qvalue_1, tens_target_qvalue_2)
            
            tens_target_qvalue = tens_reward+(self.gamma*tens_target_qvalue)*tens_done.detach()


            tens_qvalue_1 = self.critic_1(tens_ob, tens_action.float()).squeeze()
            critic_1_loss = F.mse_loss(tens_qvalue_1, tens_target_qvalue)
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward(retain_graph=True)
            self.critic_1_optimizer.step()

            tens_qvalue_2 = self.critic_2(tens_ob, tens_action.float()).squeeze()
            critic_2_loss = F.mse_loss(tens_qvalue_2, tens_target_qvalue)
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()
            
            if(i%hyperParams.POLICY_DELAY == 0):
                
                actor_loss = -self.critic_1(tens_ob, self.actor(tens_ob)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                    target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )

                for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
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
            

