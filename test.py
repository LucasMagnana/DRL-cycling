import argparse
import sys

import torch

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

import pickle

import datetime as dt

import torch

from python.NeuralNetworks import *




if __name__ == '__main__':

    module = "CartPole-v1" #"LunarLanderContinuous-v2"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default=module, help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    
    env = wrappers.Monitor(env, directory='./videos/tests/', force=True)

    with open('./trained_networks/'+module+'.hp', 'rb') as infile:
        hyperParams = pickle.load(infile)

    if("Continuous" in module):
        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], hyperParams, tanh=1)
    else:
        actor = Actor(env.observation_space.shape[0], env.action_space.n, hyperParams)
    actor.load_state_dict(torch.load('./trained_networks/'+module+'.n'))
    actor.eval()

    MAX_STEPS = 1000

    reward = 0
    done = False

    sum_reward = 0


    ob = env.reset()
    reward_accumulee=0
    steps=0
    while True:
        tens_action = actor(torch.tensor(ob, dtype=torch.float32)).data
        if("Continuous" in module):
            action = tens_action.numpy()
        else:
            _, indices = tens_action.max(0)
            action = indices.item()
        ob, reward, done, _ = env.step(action)
        reward_accumulee += reward
        steps+=1
        if done or steps > MAX_STEPS:
            break

    print("Reward: ", reward_accumulee)

    # Close the env and write monitor result info to disk
    env.close()