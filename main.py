import argparse
import sys

import matplotlib.pyplot as plt

import datetime as dt

from python.Agent import *
from python.constantes import *
from python.Environment import *





if __name__ == '__main__':

    cuda = torch.cuda.is_available()

    module =  "monresovelo"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default=module, help='Select the environment to run')
    args = parser.parse_args()
    
    env = Environment(module)

    
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
            action = agent.act(ob, reward, done).numpy()
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            steps+=1
            if done or steps > MAX_STEPS:
                if(len(agent.buffer)>LEARNING_START):
                    agent.learn(steps)
                reward_accumulee = env.overlap
                tab_rewards_accumulees.append(reward_accumulee)
                if(nb_reward < 100):
                    nb_reward+=1
                else:
                    sum_reward -= tab_rewards_accumulees[len(tab_rewards_accumulees)-nb_reward+1]
                sum_reward += reward_accumulee
                avg_reward = sum_reward/nb_reward
                break
            if(reward_accumulee > 1):
                print("error reward_accumulee")
                break
          
    print("end:", dt.datetime.now())

    print("Average: ", avg_reward)


    plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    
    print("Saving...")
    torch.save(agent.actor_target.state_dict(), './trained_networks/'+module+'_target.n')
    torch.save(agent.actor.state_dict(), './trained_networks/'+module+'.n')
        
    plt.savefig("./images/"+module+".png")

    # Close the env and write monitor result info to disk
    env.close()