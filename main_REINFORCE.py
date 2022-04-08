import argparse
import sys
import pickle
import matplotlib.pyplot as plt

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from python.hyperParams import REINFORCEHyperParams, action_space, observation_space, width, height, num_agents
from python.MesaModel import *
from python.REINFORCEAgent import *

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal


if __name__ == '__main__':

    testing = False

    hyperParams = REINFORCEHyperParams()

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "--test"):
            testing = True

    waiting_dict = {}
    for i in range(width):
        for j in range(height):
            waiting_dict[(i, j)] = 3 #randint(hyperParams.RANGE_STEP_TO_WAIT[0], hyperParams.RANGE_STEP_TO_WAIT[1])

    reinforce_agent = REINFORCEAgent(action_space, observation_space, hyperParams)
    env = MesaModel(num_agents, width, height, waiting_dict, reinforce_agent, hyperParams, testing)

    if(not testing):

        for ep in range(1, hyperParams.EPISODE_COUNT+1):

            env = MesaModel(num_agents, width, height, waiting_dict, reinforce_agent, hyperParams, testing)
            reinforce_agent.act(env, render=testing)

            if(ep%hyperParams.BATCH_SIZE):
                reinforce_agent.learn()
                reinforce_agent.reset_batches()
                
        print()

        #plot the sums of rewards and the noise (noise shouldnt be in the same graph but for now it's good)
        plt.figure(figsize=(25, 12), dpi=80)
        plt.plot(reinforce_agent.total_rewards, linewidth=1)
        plt.plot(reinforce_agent.avg_rewards, linewidth=1)
        plt.ylabel('Sum of the rewards')       
        plt.savefig("./images/mesa_REINFORCE.png")
        
        #save the neural networks of the policy
        #print(reinforce_agent.old_actor.state_dict())
        torch.save(reinforce_agent.actor.state_dict(), './trained_networks/mesa_REINFORCE.n')

        #save the hyper parameters (for the tests and just in case)
        with open('./trained_networks/mesa_REINFORCE.hp', 'wb') as outfile:
            pickle.dump(hyperParams, outfile)


    else:
        with open('./trained_networks/mesa_REINFORCE.hp', 'rb') as infile:
            hyperParams = pickle.load(infile)

        actor_to_load='./trained_networks/mesa_REINFORCE.n'
        reinforce_agent = REINFORCEAgent(action_space, observation_space, hyperParams, actor_to_load)
        params = {"N":num_agents, "width": width, "height": height, "waiting_dict": waiting_dict, "decision_maker": reinforce_agent, "hyperParams": hyperParams, "testing": testing}
        grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
        server = ModularServer(MesaModel,
                            [grid],
                            "Mesa Model",
                            params)
        server.port = 8521 # The default
        server.launch()
                