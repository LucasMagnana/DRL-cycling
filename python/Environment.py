import pickle
import numpy as np
import networkx as nx

import python.metric as metric

class ActionSpace:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.low = low
        self.high = high
        
        
class ObservationSpace:
    def __init__(self, shape):
        self.shape = shape
        

class Environment:
    
    def __init__(self, project_folder):       
        with open("files/"+project_folder+"/city_graphs/city.ox",'rb') as infile:
            self.G = pickle.load(infile)
            
        with open("files/"+project_folder+"/data_processed/observations_matched.tab",'rb') as infile:
            self.observations = pickle.load(infile)
        
        print(self.compute_overlap())
            
        self.action_space = ActionSpace([len(self.G.edges)], [-1], [1])
        self.observation_space = ObservationSpace([len(self.G.edges)])
        
        self.threshold = 0.99
        
    def compute_overlap(self):
        overlap = 0
        for ob in self.observations:
            sp = nx.shortest_path(self.G, source=ob[0], target=ob[-1], weight="w")
            m = metric.get_overlap(sp, ob, self.G, weight="w")
            if(m>1):
                 metric.get_overlap(sp, ob, self.G, weight="w", debug=True)
            overlap += m
            
        overlap/=len(self.observations)
        if(overlap > 1):
            print("error compute_overlap")
        return overlap
        
        
        
    def reset(self):
        self.observation = []
        for e in self.G.edges():
            self.G[e[0]][e[1]][0]['w'] = 1
            self.observation.append(1)
        self.overlap = self.compute_overlap()
        self.observation = np.array(self.observation)
        return self.observation
    
    def step(self, action):
        self.observation = []
        i = 0
        for e in self.G.edges():
            if(self.G[e[0]][e[1]][0]['w'] > abs(action[i])):
                self.G[e[0]][e[1]][0]['w'] += action[i]
            self.observation.append(self.G[e[0]][e[1]][0]['w'])
            i+=1
        self.observation = np.array(self.observation)
        ov = self.compute_overlap()
        reward = ov - self.overlap
        self.overlap = ov
        return self.observation, reward, self.overlap > self.threshold, None
        
        
