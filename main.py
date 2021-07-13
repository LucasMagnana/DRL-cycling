import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

import datetime as dt

from python.Agent import *
from python.constantes import *





if __name__ == '__main__':

    cuda = torch.cuda.is_available()

    module = "MountainCarContinuous-v0" #"LunarLanderContinuous-v2"
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
    
    outdir = './videos/'+module
    env = wrappers.Monitor(env, directory=outdir, video_callable=None, force=True)
    env.seed(0)

    agent = Agent(env.action_space, env.observation_space, cuda)

    reward = 0
    done = False


    tab_rewards_accumulees = []

    sum_reward = 0
    nb_reward = 0
    avg_reward = 0
    nb_episodes = 0

    save = False
    
    print("start:", dt.datetime.now())

    for e in range(1, EPISODE_COUNT): #for i in range(episode_count):
        if(e%(EPISODE_COUNT//4) == 0):
            print("1/4:", dt.datetime.now())
        ob = env.reset()
        reward_accumulee=0
        steps=0
        while True:
            ob_prec = ob  
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action[0], ob, reward, done)
            reward_accumulee += reward
            if(len(agent.buffer)>LEARNING_START):
                agent.learn()
            steps+=1
            if done or steps > MAX_STEPS:
                tab_rewards_accumulees.append(reward_accumulee)
                if(nb_reward < 100):
                    nb_reward+=1
                else:
                    sum_reward -= tab_rewards_accumulees[len(tab_rewards_accumulees)-nb_reward+1]
                sum_reward += reward_accumulee
                avg_reward = sum_reward/nb_reward
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for 
            
    print("end:", dt.datetime.now())

    print("Average: ", avg_reward)


    plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    
    if(avg_reward > 89):
        print("Saving...")
        torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'.n')
        plt.savefig("./images/"+module+".png")

    # Close the env and write monitor result info to disk
    env.close()