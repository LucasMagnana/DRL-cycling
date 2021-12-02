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
        env = DiscreteEnvironment(module)
    else:
        env = gym.make(module)

    if("Continuous" in module):      
        agent = ContinuousAgent(env.action_space, env.observation_space, cuda)
    else:
        agent = DiscreteAgent(env.action_space, env.observation_space, cuda)

    reward = 0
    done = False


    tab_rewards_accumulees = []
    tab_noise = []

    sum_reward = 0
    nb_reward = 0
    avg_reward = 0
    nb_episodes = 0

    episode_count = hyperParams.EPISODE_COUNT
    max_steps = hyperParams.MAX_STEPS
    
    print("start:", dt.datetime.now())

    for e in range(1, episode_count): #for i in range(episode_count):
        if(e%(episode_count//4) == 0):
            print("1/4:", dt.datetime.now())
        ob = env.reset()
        reward_accumulee=0
        steps=0
        while True:
            ob_prec = ob  
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            steps+=1
            if done or steps > max_steps:
                if(len(agent.buffer)>hyperParams.LEARNING_START):
                    agent.learn(steps)
                tab_rewards_accumulees.append(reward_accumulee)
                if(agent.epsilon > 0):
                    tab_noise.append(agent.epsilon)               
                if(nb_reward < 100):
                    nb_reward+=1
                else:
                    sum_reward -= tab_rewards_accumulees[len(tab_rewards_accumulees)-nb_reward+1]
                sum_reward += reward_accumulee
                avg_reward = sum_reward/nb_reward
                break
          
    print("end:", dt.datetime.now())

    #print("Average: ", avg_reward)

    plt.figure(figsize=(25, 12), dpi=80)
    plt.plot(tab_rewards_accumulees, linewidth=1)
    plt.plot(tab_noise)
    plt.ylabel('Reward Accumulée')       
    plt.savefig("./images/"+module+".png")
    
    print("Saving...")
    torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), './trained_networks/'+module+'.n')

    with open('./trained_networks/'+module+'.hp', 'wb') as outfile:
        pickle.dump(hyperParams, outfile)


    if("monresovelo" in module):
        test()


    # Close the env and write monitor result info to disk
    env.close()