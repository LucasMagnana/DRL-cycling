from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.batchrunner import FixedBatchRunner
import matplotlib.pyplot as plt
import pickle
from random import *

from python.MesaModel import *
import python.hyperParams

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal

list_rewards = []

testing = True

cuda=True
print("GPU:", cuda)
if(cuda):
    print(torch.cuda.get_device_name(0))

width=4
height=7
waiting_dict = {}
for i in range(width):
    for j in range(height):
        waiting_dict[(i, j)] = 3 #randint(hyperParams.RANGE_STEP_TO_WAIT[0], hyperParams.RANGE_STEP_TO_WAIT[1])


decision_maker = DiscreteAgent(DiscreteActionSpace(5), DiscreteObservationSpace(8), cuda=cuda)
params = {"N":2, "width":width, "height":height, "waiting_dict":waiting_dict, "decision_maker": decision_maker, "list_rewards": list_rewards, "testing":testing}

if(testing):
    with open('./trained_networks/mesa.hp', 'rb') as infile:
        hyperParams = pickle.load(infile)
    with open('./trained_networks/waiting.dict', 'rb') as infile:
        waiting_dict = pickle.load(infile)
    params["decision_maker"] = DiscreteAgent(DiscreteActionSpace(5), DiscreteObservationSpace(8), cuda=cuda, hyperParams=hyperParams, actor_to_load='./trained_networks/mesa.n')
    params["waiting_dict"] = waiting_dict
    grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
    server = ModularServer(MesaModel,
                        [grid],
                        "Mesa Model",
                        params)
    server.port = 8521 # The default
    server.launch()

else:
    batch_runner = FixedBatchRunner(MesaModel, fixed_parameters=params,
    iterations=hyperParams.EPISODE_COUNT, max_steps=hyperParams.MAX_STEPS)
    batch_runner.run_all()

    #save the neural networks of the decision maker
    print("Saving...")
    torch.save(decision_maker.actor_target.state_dict(), './trained_networks/mesa_target.n')
    torch.save(decision_maker.actor.state_dict(), './trained_networks/mesa.n')

    #save the hyper parameters (for the tests and just in case)
    with open('./trained_networks/mesa.hp', 'wb') as outfile:
        pickle.dump(hyperParams, outfile)

    with open('./trained_networks/waiting.dict', 'wb') as outfile:
        pickle.dump(waiting_dict, outfile)

    plt.figure(figsize=(25, 12), dpi=80)
    plt.plot(list_rewards, linewidth=1)
    plt.ylabel('Reward Accumulée')       
    plt.savefig("./images/mesa.png")