import argparse
import sys

import torch

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

import datetime as dt

import torch

from python.NeuralNetworks import *




if __name__ == '__main__':

    MAX_STEPS = 1000

    module = "LunarLanderContinuous-v2"
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

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    actor.load_state_dict(torch.load('./trained_networks/'+module+'.n'))
    actor.eval()

    reward = 0
    done = False

    sum_reward = 0


    ob = env.reset()
    reward_accumulee=0
    steps=0
    while True:
        action = actor(torch.tensor(ob, dtype=torch.float32)).data.numpy()
        ob, reward, done, _ = env.step(action)
        reward_accumulee += reward
        steps+=1
        if done or steps > MAX_STEPS:
            break

    print("Reward: ", reward_accumulee)

    # Close the env and write monitor result info to disk
    env.close()