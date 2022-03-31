
import pickle
import torch
import numpy as np
import copy

from python.NeuralNetworks import PPO_Actor, PPO_Critic

def discount_rewards(rewards, gamma):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages


class PPOAgent():

    def __init__(self, ac_space, ob_space, hyperParams, actor_to_load=None, critic_to_load=None):
        self.action_space = ac_space
        self.observation_space = ob_space

        self.hyperParams = hyperParams
        self.old_actor = PPO_Actor(self.observation_space.shape[0], self.action_space.n, self.hyperParams)
        self.critic = PPO_Critic(self.observation_space.shape[0], self.hyperParams)

        if(actor_to_load != None and critic_to_load != None):
            self.old_actor.load_state_dict(torch.load(actor_to_load))
            self.old_actor.eval()
            #print(self.old_actor.state_dict())
            self.critic.load_state_dict(torch.load(critic_to_load))
            self.critic.eval()

            self.actor = copy.deepcopy(self.old_actor)   
            # Define optimizer
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hyperParams.LR)
            self.mse = torch.nn.MSELoss()
        else:
            self.actor = copy.deepcopy(self.old_actor)   
            # Define optimizer
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.hyperParams.LR)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.hyperParams.LR)
            self.mse = torch.nn.MSELoss()

        self.avg_rewards = []
        self.total_rewards = []

        self.ep = 0

        self.reset_batches()


    def reset_batches(self):
        self.batch_rewards = []
        self.batch_advantages = []
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_done = []


    def learn(self):

        for k in range(self.hyperParams.K):
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            state_tensor = torch.tensor(self.batch_states)
            advantages_tensor = torch.tensor(self.batch_advantages)

            values_tensor = torch.tensor(self.batch_values)
            rewards_tensor = torch.tensor(self.batch_rewards, requires_grad = True)
            rewards_tensor = rewards_tensor.float()
            # Actions are used as indices, must be 
            # LongTensor
            action_tensor = torch.LongTensor(self.batch_actions)
            action_tensor = action_tensor.long()

            
            # Calculate actor loss
            probs = self.actor(state_tensor)
            selected_probs = torch.index_select(probs, 1, action_tensor).diag()

            old_probs = self.old_actor(state_tensor)
            selected_old_probs = torch.index_select(old_probs, 1, action_tensor).diag()

            loss = selected_probs/selected_old_probs*advantages_tensor
            clipped_loss = torch.clamp(selected_probs/selected_old_probs, 1-self.hyperParams.EPSILON, 1+self.hyperParams.EPSILON)*advantages_tensor

            loss_actor = -torch.min(loss, clipped_loss).mean()



            # Calculate gradients
            loss_actor.backward()
            # Apply gradients
            self.optimizer_actor.step()

            # Calculate critic loss
            loss_critic = self.mse(values_tensor, rewards_tensor)  
            # Calculate gradients
            loss_critic.backward()
            # Apply gradients
            self.optimizer_critic.step()

        self.old_actor = copy.deepcopy(self.actor)


    def act(self, env, render=False):

        for y in range(self.hyperParams.NUM_EP_ENV):
            while(env.running):
                env.step()       

            for key in env.states:
                if(not(len(env.states[key]) == len(env.rewards[key]) == len(env.actions[key]) == len(env.values[key]) == len(env.list_done[key]))):
                    print("GROSSE BITE !!!!!!")
                self.batch_rewards.extend(discount_rewards(env.rewards[key], self.hyperParams.GAMMA))
                gaes = gae(np.expand_dims(np.array(env.rewards[key]), 0), np.expand_dims(np.array(env.values[key]), 0), np.expand_dims(np.array([not elem for elem in env.list_done[key]]), 0), self.hyperParams.GAMMA, self.hyperParams.LAMBDA)
                self.batch_advantages.extend(gaes[0])
                self.batch_states.extend(env.states[key])
                self.batch_values.extend(env.values[key])
                self.batch_actions.extend(env.actions[key])
               
                self.batch_done.extend(env.list_done[key])
            
            self.total_rewards.append(env.mean_reward)
            ar = np.mean(self.total_rewards[-100:])
            self.avg_rewards.append(ar)

            self.ep += 1

            if(not render):
                print("\rEp: {} Average of last 100: {:.2f}".format(self.ep, ar), end="")



                