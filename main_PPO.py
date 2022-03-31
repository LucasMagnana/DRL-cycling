import argparse
import sys
import pickle
import matplotlib.pyplot as plt

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from python.hyperParams import PPOHyperParams
from python.MesaModel import *
from python.PPOAgent import *

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal


if __name__ == '__main__':

    testing = False

    hyperParams = PPOHyperParams()

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True

    num_agents = 2

    width=4
    height=7
    waiting_dict = {}
    for i in range(width):
        for j in range(height):
            waiting_dict[(i, j)] = 3 #randint(hyperParams.RANGE_STEP_TO_WAIT[0], hyperParams.RANGE_STEP_TO_WAIT[1])

    ppo_agent = PPOAgent(DiscreteActionSpace(5), DiscreteObservationSpace(4), hyperParams, './trained_networks/mesa_ac_PPO.n', './trained_networks/mesa_cr_PPO.n')
    env = MesaModel(num_agents, width, height, waiting_dict, ppo_agent, hyperParams, testing)

    if(not testing):

        for ep in range(hyperParams.EPISODE_COUNT):

            ppo_agent.reset_batches()


            for a in range(hyperParams.NUM_AGENTS):
                env = MesaModel(num_agents, width, height, waiting_dict, ppo_agent, hyperParams, testing)
                ppo_agent.act(env, render=testing)

            ppo_agent.learn()
                
        print()

        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(ppo_agent.total_rewards, linewidth=1)
        plt.plot(ppo_agent.avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/mesa_PPO.png")
        
        #save the neural networks of the policy
        #print(ppo_agent.old_actor.state_dict())
        torch.save(ppo_agent.old_actor.state_dict(), './trained_networks/mesa_ac_PPO.n')
        torch.save(ppo_agent.critic.state_dict(), './trained_networks/mesa_cr_PPO.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/mesa_PPO.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)


    else:
        with open('./trained_networks/mesa_PPO.hp', 'rb') as infile:
            hyperParams = pickle.load(infile)

        actor_to_load='./trained_networks/mesa_ac_PPO.n'
        critic_to_load='./trained_networks/mesa_cr_PPO.n'
        decision_maker = PPOAgent(DiscreteActionSpace(5), DiscreteObservationSpace(4), hyperParams, actor_to_load, critic_to_load)
        params = {"N":2, "width": width, "height": height, "waiting_dict": waiting_dict, "decision_maker": ppo_agent, "hyperParams": hyperParams, "testing": testing}
        grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
        server = ModularServer(MesaModel,
                            [grid],
                            "Mesa Model",
                            params)
        server.port = 8521 # The default
        server.launch()
                