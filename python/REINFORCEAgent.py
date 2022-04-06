
import pickle
import torch
import numpy as np
import copy

from python.NeuralNetworks import REINFORCE_Model

def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()



class REINFORCEAgent():

    def __init__(self, ac_space, ob_space, hyperParams, actor_to_load=None):
        self.action_space = ac_space
        self.observation_space = ob_space

        self.hyperParams = hyperParams
        self.actor = REINFORCE_Model(self.observation_space.shape[0], self.action_space.n, self.hyperParams)

        if(actor_to_load != None):
            self.actor.load_state_dict(torch.load(actor_to_load))
            self.actor.eval()

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR)

        self.avg_rewards = []
        self.total_rewards = []

        self.ep = 0

        self.reset_batches()


    def reset_batches(self):
        self.batch_rewards = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_number = 0


    def learn(self):

        self.optimizer.zero_grad()
        state_tensor = torch.tensor(self.batch_states)
        reward_tensor = torch.tensor(self.batch_rewards)
        # Actions are used as indices, must be 
        # LongTensor
        action_tensor = torch.LongTensor(self.batch_actions)
        action_tensor = action_tensor.long()
        
        # Calculate loss
        logprob = torch.log(self.actor(state_tensor))
        selected_logprobs = reward_tensor * torch.index_select(logprob, 1, action_tensor).diag()
        loss = -selected_logprobs.mean()
        '''print()
        print(loss.item())'''
        
        # Calculate gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        # Apply gradients
        self.optimizer.step()



    def act(self, env, render=False):

        while(env.running):
            env.step()       

        for key in env.states:

            self.batch_rewards.extend(discount_rewards(env.rewards[key], self.hyperParams.GAMMA))
            self.batch_states.extend(env.states[key])
            self.batch_actions.extend(env.actions[key])
        
        self.total_rewards.append(env.mean_reward)
        ar = np.mean(self.total_rewards[-100:])
        self.avg_rewards.append(ar)

        self.ep += 1

        if(not render):
            print("\rEp: {} Average of last 100: {:.2f}".format(self.ep, ar), end="")


    def choose_action(self, ob, testing):
        action_probs = self.actor(torch.tensor(ob)).detach().numpy()
        #print(action_probs, ob)
        if(testing):
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(np.arange(self.action_space.n), p=action_probs)
        return [None], action



                