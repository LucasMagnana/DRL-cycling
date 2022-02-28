from random import sample
from random import * 
from torch.nn import MSELoss

import numpy as np

import copy
import numpy as np
import torch

from python.hyperParams import hyperParams as hp

from python.NeuralNetworks import Actor, ActorRNN

class DiscreteAgent(object):
    def __init__(self, action_space, observation_space, cuda, hyperParams=None, actor_to_load=None):

        if(hyperParams == None): #use the good hyper parameters (loaded if it's a test, written in the code if it's a training)
            self.hyperParams = hp
        else:
            self.hyperParams = hyperParams
            self.hyperParams.EPSILON = 0

        self.action_space = action_space   

        self.buffer = [] #replay buffer of the agent
        self.buffer_max_size = self.hyperParams.BUFFER_SIZE

        self.alpha = self.hyperParams.ALPHA
        self.epsilon = self.hyperParams.EPSILON
        self.gamma = self.hyperParams.GAMMA

        self.actor = ActorRNN(observation_space.shape[0], action_space.n) #for custom env
        #self.actor = Actor(observation_space.shape[0], action_space.n) #for cartpole

        self.batch_size = self.hyperParams.BATCH_SIZE


        if(actor_to_load != None): #if it's a test, use the loaded NN
            self.actor.load_state_dict(torch.load(actor_to_load))
            self.actor.eval()
        
        self.actor_target = copy.deepcopy(self.actor) #a target network is used to make the convergence possible (see papers on DRL)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), self.hyperParams.LR) # smooth gradient descent

        


    def act(self, observation):
        #return self.action_space.sample()
        tens_qvalue = self.actor(torch.Tensor(observation).unsqueeze(0)) #compute the qvalues for the observation
        tens_qvalue = tens_qvalue.squeeze()
        rand = random()
        if(rand > self.epsilon): #noise management
            _, indices = tens_qvalue.max(0) #finds the index of the max qvalue
            return indices.item() #return it
        return randint(0, tens_qvalue.size()[0]-1) #choose a random action

    def sample(self):
        if(len(self.buffer) < self.batch_size):
            return sample(self.buffer, len(self.buffer))
        else:
            return sample(self.buffer, self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_max_size): #delete the first element if the buffer is at max size 
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self, n_iter): #n_iter only used in continuousAgent, not here

        #previous noise decaying method, works well with cartpole
        '''if(self.epsilon > self.hyperParams.MIN_EPSILON):
            self.epsilon *= self.hyperParams.EPSILON_DECAY
        else:
            self.epsilon = 0'''

        
        #actual noise decaying method, works well with the custom env
        self.epsilon -= self.hyperParams.EPSILON_DECAY

        loss = MSELoss()

        spl = self.sample()  #create a batch of experiences

        tens_state = torch.Tensor([item[0] for item in spl]) #get all the actual states
        tens_action = torch.LongTensor([item[1] for item in spl]) #get all the actions chosen by the agent
        tens_state_next = torch.Tensor([item[2] for item in spl]) #get all the states sent by the env after the actions
        tens_reward = torch.Tensor([item[3] for item in spl]) #get all the rewards
        tens_done = torch.Tensor([item[4] for item in spl]) #for each experience, get if the state after the action is final

        tens_qvalue = self.actor(tens_state) #compute the qvalues for all the actual states
        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag() #select the qvalues corresponding to the chosen actions

        if(self.hyperParams.DOUBLE_DQN == False):
            # Simple DQN
            tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states"
            (tens_next_qvalue, _) = torch.max(tens_next_qvalue, 1) #select the max qvalues for all the next states
        else:
            # Double DQN
            tens_next_qvalue = self.actor(tens_state_next) #compute all the qvalues for all the "next states" with the ppal network
            (_, tens_next_action) = torch.max(tens_next_qvalue, 1) #returns the indices of the max qvalues for all the next states(to choose the next actions)
            tens_next_qvalue = self.actor_target(tens_state_next) #compute all the qvalues for all the "next states" with the target network
            tens_next_qvalue = torch.index_select(tens_next_qvalue, 1, tens_next_action).diag() #select the qvalues corresponding to the chosen next actions

        self.optimizer.zero_grad() #reset the gradient
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done) #calculate the loss
        tens_loss.backward() #compute the gradient
        self.optimizer.step() #back-propagate the gradient

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()): #updates the target network
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )