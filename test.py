import argparse
import sys
import gym 
import pickle

import matplotlib.pyplot as plt

import datetime as dt

from python.ContinuousAgent import *
from python.DiscreteAgent import *
from python.hyperParams import hyperParams, module
import python.metric as metric

from python.DiscreteEnvironment import *
from python.ContinuousEnvironment import *





def test(module="monresovelo"):

    cuda = torch.cuda.is_available()
    
    env = DiscreteEnvironment(module)

    with open('./trained_networks/'+module+'.hp', 'rb') as infile:
        hyperParams = pickle.load(infile)
    
    agent = DiscreteAgent(env.action_space, env.observation_space, cuda, hyperParams=hyperParams, actor_to_load='./trained_networks/'+module+'.n')

    sum_overlap = 0
    sum_reward = 0

    for path in env.list_paths:
        reward = 0
        done = False
        ob = env.reset(path)
        reward_accumulee=0
        steps=0
        while True:
            ob_prec = ob  
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            steps+=1
            if done or steps > hyperParams.MAX_STEPS:
                original_path = []
                for e in path:
                    original_path.append(e[0])
                original_path.append(path[-1][-1])
                
                generated_path = []
                for n in env.generated_path:
                    generated_path.append(n[0])

                sum_overlap += metric.get_overlap(generated_path, original_path, env.G)
                sum_reward += reward_accumulee
                break

    print("Average reward : ", sum_reward/len(env.list_paths))   
    print("Average overlap : ", sum_overlap/len(env.list_paths))


if __name__ == '__main__':
    test()