import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.ContinuousAgent import *
from python.DiscreteAgent import *
from python.hyperParams import hyperParams, module

from python.DiscreteEnvironment import *
from python.ContinuousEnvironment import *

from test import test





if __name__ == '__main__':

    cuda = torch.cuda.is_available()

    env = None
    
    if("monresovelo" in module):
        env = DiscreteEnvironment(module) #custom env
    else:
        env = gym.make(module) #gym env

    if("Continuous" in module): #agents are not the same wether the action space is continuous or discrete     
        agent = ContinuousAgent(env.action_space, env.observation_space, cuda)
    else:
        agent = DiscreteAgent(env.action_space, env.observation_space, cuda=cuda)

    tab_sum_rewards = []
    tab_noise = []
    
    print("start:", dt.datetime.now())

    for e in range(1, hyperParams.EPISODE_COUNT):

        if(e%(hyperParams.EPISODE_COUNT//4) == 0):
            print("1/4:", dt.datetime.now())

        ob = env.reset()
        sum_rewards=0
        steps=0
        while True:
            ob_prec = ob   
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            sum_rewards += reward
            steps+=1
            if(len(agent.buffer)>hyperParams.LEARNING_START):
                agent.learn(steps)
            if done or steps > hyperParams.MAX_STEPS:
                tab_sum_rewards.append(sum_rewards)
                if(agent.epsilon > 0):
                    tab_noise.append(agent.epsilon)               
                break
          
    print("end:", dt.datetime.now())

    
    #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
    plt.figure(figsize=(25, 12), dpi=80)
    plt.plot(tab_sum_rewards, linewidth=1)
    plt.plot(tab_noise)
    plt.ylabel('Reward Accumulée')       
    plt.savefig("./images/"+module+".png")

    avg_last_sum_rewards = sum(tab_sum_rewards[-100:])/100

    print("Average last 100 sums of reward:", avg_last_sum_rewards)
    
    #save the neural networks of the agent
    print("Saving...")
    torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), './trained_networks/'+module+'.n')

    #save the hyper parameters (for the tests and just in case)
    with open('./trained_networks/'+module+'.hp', 'wb') as outfile:
        pickle.dump(hyperParams, outfile)


    if("monresovelo" in module): #tests only work for the custom env (so will the project in the end)
        test()


    # Close the env (only useful for the gym envs for now)
    env.close()