from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from random import randint
import torch 
import numpy as np
import copy


class DiscreteActionSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, n):
        self.n = n
        
        
class DiscreteObservationSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, shape):
        self.shape = [shape]


class MesaAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, hyperParams):
        super().__init__(unique_id, model)

        self.hyperParams = hyperParams

        self.shortest_path = None
        self.next_position = None

        self.estimated_ending_step = None

        self.in_group = False
        self.reduction_step_applied = False

        self.moved = False

        self.n_step = 0

        self.sum_reward = 0
        self.ob = None
        self.ob_prec = None
        self.action = None
        self.value = None

        self.id = unique_id



    def compute_shortest_path(self):
        if(self.next_position == None):
            self.shortest_path = [self.pos]
            self.path_taken = [self.pos]
        else:
            self.shortest_path = [self.next_position]

        while(self.shortest_path[-1][0] != self.destination[0]):
            if(self.shortest_path[-1][0] < self.destination[0]):
                self.shortest_path.append((self.shortest_path[-1][0]+1, self.shortest_path[-1][1]))
            else:
                self.shortest_path.append((self.shortest_path[-1][0]-1, self.shortest_path[-1][1]))

        while(self.shortest_path[-1][1] != self.destination[1]):
            if(self.shortest_path[-1][1] < self.destination[1]):
                self.shortest_path.append((self.shortest_path[-1][0], self.shortest_path[-1][1]+1))
            else:
                self.shortest_path.append((self.shortest_path[-1][0], self.shortest_path[-1][1]-1))

        self.shortest_path.pop(0)

        if(self.estimated_ending_step == None):
            self.next_movement_step = self.n_step + self.model.waiting_dict[(self.pos)]
            self.estimated_ending_step = self.model.waiting_dict[(self.pos)]
            for p in self.shortest_path:
                self.estimated_ending_step += self.model.waiting_dict[p]


    def change_next_position_with_action(self):
        if(self.action==0):
            if(self.pos[0] == self.model.grid.width-1):
                self.next_position = (self.pos[0]-1, self.pos[1])
            else:
                self.next_position = (self.pos[0]+1, self.pos[1])
        elif(self.action==1):
            if(self.pos[0] == 0):
                self.next_position = (self.pos[0]+1, self.pos[1])
            else:
                self.next_position = (self.pos[0]-1, self.pos[1])
        elif(self.action==2):
            if(self.pos[1] == self.model.grid.height-1):
                self.next_position = (self.pos[0], self.pos[1]-1)
            else:
                self.next_position = (self.pos[0], self.pos[1]+1)
        elif(self.action==3):
            if(self.pos[1] == 0):
                self.next_position = (self.pos[0], self.pos[1]+1)
            else:
                self.next_position = (self.pos[0], self.pos[1]-1)
        else:
            self.next_position = (self.pos[0], self.pos[1])

        self.compute_shortest_path()


    def get_padded_path_taken(self):
        padded_path_taken = copy.deepcopy(self.path_taken)
        for i in range(len(padded_path_taken)):
            padded_path_taken[i] = (padded_path_taken[i][0]+1, padded_path_taken[i][1]+1)
        for _ in range(self.hyperParams.SEQ_SIZE-len(self.path_taken)):
            padded_path_taken.append((0,0))
        return padded_path_taken


    def construct_and_save_observation(self):
        observation = [self.pos[0]+1, self.pos[1]+1, self.destination[0]+1, self.destination[1]+1]

        '''for n in self.model.grid.get_neighbors(self.pos, True, radius=5):
            if(n != self):
                neighbor_observation = [n.pos[0]+1, n.pos[1]+1, n.destination[0]+1, n.destination[1]+1]
                if(len(observation)<self.model.decision_maker.observation_space.shape[0]):
                    observation += neighbor_observation

        while(len(observation)<self.model.decision_maker.observation_space.shape[0]):
            observation.append(0)'''
        #print(observation)
        self.ob_prec = self.ob
        self.ob = observation #[self.get_padded_path_taken(), observation]


    def calculate_reward(self):
        done = False
        if(self.pos == self.destination):
            done = True
        if(self.estimated_ending_step < self.n_step):
            r = -1
        elif(self.in_group):
            r = 1
        elif(not self.in_group):
            r = 0

        self.sum_reward += r
        return done, r



    def move(self):
        if(self.shortest_path == None):
            self.compute_shortest_path()
            self.next_position = self.shortest_path[0]
        
        if(self.in_group and not self.reduction_step_applied):
            self.next_movement_step = self.next_movement_step - 2
            self.reduction_step_applied = True


        if(self.n_step >= self.next_movement_step):
            if(self.ob == None):
                self.construct_and_save_observation()
            action_probs = self.model.decision_maker.old_actor(torch.tensor(self.ob)).detach().numpy()
            self.action = np.random.choice(np.arange(self.model.decision_maker.action_space.n), p=action_probs)
            self.value = self.model.decision_maker.critic(torch.tensor(self.ob)).detach().numpy()
            self.change_next_position_with_action()
            self.model.grid.move_agent(self, self.next_position)
            self.path_taken.append(self.next_position)
            self.moved = True

            if(self.pos == self.destination):
                self.next_position = self.pos
                #print(self.n_step, self.estimated_ending_step)
            else:
                self.compute_shortest_path()    
                self.next_position = self.shortest_path[0]
                self.shortest_path.pop(0)
                self.next_movement_step = self.n_step + self.model.waiting_dict[(self.pos)]
                self.reduction_step_applied = False

        


        

    def step(self):
        self.n_step+=1
        self.move()


class MesaModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, waiting_dict, decision_maker, hyperParams, testing):
        super().__init__()

        self.num_agents = N
        self.waiting_dict = waiting_dict
        self.decision_maker = decision_maker
        self.testing = testing
        self.hyperParams = hyperParams

        self.grid = MultiGrid(width, height, True)
        
        self.list_agents = []

        self.reset()
        

    def reset(self):
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            a = MesaAgent(i, self, self.hyperParams)
            self.schedule.add(a)

            if(self.testing):            
                if(i == 0):
                    self.grid.place_agent(a, (0, 1))
                    a.destination = ((self.grid.width-1)//2, self.grid.height-1)
                else:
                    self.grid.place_agent(a, (self.grid.width-1, 0))
                    a.destination = ((self.grid.width-1)//2+1, self.grid.height-1)

            else:              
                pos=(randint(0, self.grid.width-1), randint(0, self.grid.height-1))
                self.grid.place_agent(a, pos)
                dest=pos
                while(dest==pos):
                    dest=(randint(0, self.grid.width-1), randint(0, self.grid.height-1))
                
                a.destination=dest

            self.list_agents.append(a)

        self.states = {}
        self.rewards = {}
        self.actions = {}
        self.values = {}
        self.list_done = {}

        for a in self.list_agents:
            self.states[a.id] = []
            self.rewards[a.id] = []
            self.actions[a.id] = []
            self.values[a.id] = []
            self.list_done[a.id] = []

        self.n_iter = 0
        self.mean_reward = 0
        self.running = True




    def step(self):
        """Advance the model by one step."""
        self.n_iter+=1

        next_list_agents = []
        self.schedule.step()
        self.check_for_groups()
        for a in self.list_agents:
            a.construct_and_save_observation()

        for a in self.list_agents:
            if(a.moved):
                done, reward = a.calculate_reward()
                self.states[a.id].append(a.ob_prec)
                self.values[a.id].extend(a.value)
                self.actions[a.id].append(a.action)
                self.rewards[a.id].append(reward)
                self.list_done[a.id].append(done) 
                if(done):
                    self.mean_reward += (a.sum_reward/self.num_agents)
                    self.grid.remove_agent(a)
                    self.schedule.remove(a)
                else:
                    next_list_agents.append(a)
                a.moved = False
            else:
                next_list_agents.append(a)

        self.list_agents = next_list_agents


        if(len(self.list_agents) == 0 or self.n_iter == self.hyperParams.MAX_STEPS):
            for a in self.list_agents:
                self.mean_reward += (a.sum_reward/self.num_agents)
            self.running = False

        



    def check_for_groups(self):
        for cell in self.grid.coord_iter():
            list_agents = cell[0]
            if(len(list_agents) >= self.hyperParams.MIN_NUM_AGENT_IN_GROUP):
                for a in list_agents:
                    a.in_group = True
            else:
                for a in list_agents:
                    a.in_group = False

