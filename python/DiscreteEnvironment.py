import pickle
import numpy as np
import networkx as nx
import random
import copy

from python.hyperParams import hyperParams


class DiscreteActionSpace:
    def __init__(self, n):
        self.n = n
        
        
class DiscreteObservationSpace:
    def __init__(self, shape):
        self.shape = [shape]
        

class DiscreteEnvironment:
    
    def __init__(self, project_folder):       
        with open("files/"+project_folder+"/city_graphs/city.ox",'rb') as infile:
            self.G = pickle.load(infile)
            
        with open("files/"+project_folder+"/data_processed/observations_matched.tab",'rb') as infile:
            self.list_paths = pickle.load(infile)
        
        max_number_edges = max(dict(self.G.degree()).items(), key = lambda x : x[1])[1]
        self.action_space = DiscreteActionSpace(max_number_edges)
        self.observation_space = DiscreteObservationSpace(2)

        
    def reset(self, path=None):
        if(path == None):
            self.original_path = random.choice(self.list_paths)
        else:
            self.original_path = path
        self.state = [self.original_path[0][0], self.original_path[-1][1]]
        self.generated_path = [[self.original_path[0][0]]]
        return [self.get_padded_generated_path(), self.state]

    def get_padded_generated_path(self):
        padded_generated_path = copy.deepcopy(self.generated_path)
        for _ in range(hyperParams.SEQ_SIZE-len(self.generated_path)):
            padded_generated_path.append([0])
        return padded_generated_path
    
    def step(self, action):
        next_edge = list(self.G.edges(self.state[0]))[action] #we transform the action in a chosen edge
        done = False
        
        if(next_edge[1] == self.state[-1]): #if the next node is the final node, high reward because the agent found the destination in the graph
            reward = +3
            done = True
        elif([next_edge[1]] in self.generated_path): #if the next node has already been visited, punition because we do not want loops
            reward = -3
            done = True
        elif(next_edge in self.original_path): #if the edge is used by the original path, positive reward because the agent is on the good way
            reward = +1
        elif(next_edge not in self.original_path): #if not negative value, the agent is on a wrong way
            reward = -1
        else:
            print("ERROR IN REWARD'S CALCULATION")

        self.state[0] = next_edge[1] #the actual node is now the one at the end of the chosen edge
        self.generated_path.append([self.state[0]])

        return [self.get_padded_generated_path(), self.state], reward, done, None
        
        
