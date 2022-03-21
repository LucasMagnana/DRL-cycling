from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from python.hyperParams import hyperParams
from python.DiscreteAgent import *


class DiscreteActionSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, n):
        self.n = n
        
        
class DiscreteObservationSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, shape):
        self.shape = [shape]


class MesaAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.shortest_path = None
        self.next_position = None

        self.estimated_ending_step = None

        self.in_group = False
        self.reduction_step_applied = False

        self.moved = False

        self.n_step = 0

        self.reward = 0
        self.ob = None



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


    def change_next_position(self, action):
        if(action==0):
            if(self.pos[0] == self.model.grid.width-1):
                self.next_position = (self.pos[0]-1, self.pos[1])
            else:
                self.next_position = (self.pos[0]+1, self.pos[1])
        elif(action==1):
            if(self.pos[0] == 0):
                self.next_position = (self.pos[0]+1, self.pos[1])
            else:
                self.next_position = (self.pos[0]-1, self.pos[1])
        elif(action==2):
            if(self.pos[1] == self.model.grid.height-1):
                self.next_position = (self.pos[0], self.pos[1]-1)
            else:
                self.next_position = (self.pos[0], self.pos[1]+1)
        elif(action==3):
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
        for _ in range(hyperParams.SEQ_SIZE-len(self.path_taken)):
            padded_path_taken.append((0,0))
        return padded_path_taken


    def construct_and_save_observation(self):
        observation = [self.pos[0]+1, self.pos[1]+1, self.destination[0]+1, self.destination[1]+1]

        for n in self.model.grid.get_neighbors(self.pos, True, radius=5):
            if(n != self):
                neighbor_observation = [n.pos[0]+1, n.pos[1]+1, n.destination[0]+1, n.destination[1]+1]
                if(len(observation)<self.model.decision_maker.observation_space.shape[0]):
                    observation += neighbor_observation

        while(len(observation)<self.model.decision_maker.observation_space.shape[0]):
            observation.append(0)
        #print(observation)
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

        self.reward += r
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
            self.action = self.model.decision_maker.act(self.ob)
            self.ob_prec = self.ob
            self.ob = None
            self.change_next_position(self.action)
            self.model.grid.move_agent(self, self.next_position)
            self.path_taken.append(self.next_position)
            self.moved = True

            if(self.pos == self.destination):
                self.next_position = self.pos
                #print(self.n_step, self.estimated_ending_step)
            else:
                self.next_position = self.shortest_path[0]
                self.shortest_path.pop(0)
                self.next_movement_step = self.n_step + self.model.waiting_dict[(self.pos)]
                self.reduction_step_applied = False

        


        

    def step(self):
        self.n_step+=1
        self.move()


class MesaModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, waiting_dict, decision_maker, list_rewards, testing):
        super().__init__()
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.testing = testing

        self.grid = MultiGrid(width, height, True)
        
        self.list_agents = []
        for i in range(self.num_agents):
            a = MesaAgent(i, self)
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

        self.decision_maker = decision_maker 
        self.need_learning = False

        self.n_iter = 0
        self.list_rewards = list_rewards
        self.mean_reward = 0
        self.N = N
        self.waiting_dict = waiting_dict


    def step(self):
        """Advance the model by one step."""
        self.n_iter += 1
        self.schedule.step()
        self.check_for_groups()

        next_list_agents = []

        need_learning = False

        for a in self.list_agents:
            if(a.moved):
                need_learning = True
                done, reward = a.calculate_reward()
                a.construct_and_save_observation()
                self.decision_maker.memorize(a.ob_prec, a.action, a.ob, reward, done)
                if(done):
                    self.mean_reward += (a.reward/self.N)
                if(a.pos == a.destination):
                    self.grid.remove_agent(a)
                    self.schedule.remove(a)
                else:
                    next_list_agents.append(a)
                a.moved = False
                a.ob_prec = None
                a.action = None
            else:
                next_list_agents.append(a)

        self.list_agents = next_list_agents

        if(need_learning and len(self.decision_maker.buffer) > hyperParams.LEARNING_START):
            self.decision_maker.learn()

        if(len(self.list_agents) == 0 or self.n_iter >= hyperParams.MAX_STEPS):
            for a in self.list_agents:
                self.mean_reward += (a.reward/self.N)
            self.running = False
            self.list_rewards.append(self.mean_reward)
            #print("mean :", self.mean_reward)



    def check_for_groups(self):
        for cell in self.grid.coord_iter():
            list_agents = cell[0]
            if(len(list_agents) >= hyperParams.MIN_NUM_AGENT_IN_GROUP):
                for a in list_agents:
                    a.in_group = True
            else:
                for a in list_agents:
                    a.in_group = False

