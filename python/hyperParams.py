
class DiscreteActionSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, n):
        self.n = n
        
        
class DiscreteObservationSpace: #mandatory to use an agent designed for gym environments
    def __init__(self, shape):
        self.shape = [shape]




class DDQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 100000  #max size of the agent's replays buffer
        self.ALPHA = 0.05 #
        self.GAMMA = 0.9 #discount factor
        self.LR = 0.001 #learning rate of the actor
        self.BATCH_SIZE = 256 #number of replays processed by learning step

        self.HIDDEN_SIZE_1 = 128 #size of the first hidden layer of the actor
        self.HIDDEN_SIZE_2 = 64

        self.EPISODE_COUNT = 15000 #total number of episodes
        self.MAX_STEPS = 100 #max steps by episode
        self.LEARNING_START = 1500 #number of steps before the first learning

        self.EPSILON = 1.0 #noise coefficient
        self.MIN_EPSILON = 0.05
        self.EPSILON_DECAY = self.EPSILON/(150000) #linear decay (EPS-=EPS_DECAY at each learning step)

        #specific to custom env        
        self.SEQ_SIZE = 16
        self.NUM_RNN_LAYERS = 1
        self.DOUBLE_DQN = False

        self.RANGE_STEP_TO_WAIT = [3, 9]
        self.MIN_NUM_AGENT_IN_GROUP = 2


class PPOHyperParams :
    def __init__(self):
        self.LR = 0.001
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.2

        self.EPISODE_COUNT = 600
        self.NUM_AGENTS = 10
        self.NUM_EP_ENV = 5
        self.K = 3

        self.HIDDEN_SIZE_1 = 128 #size of the first hidden layer of the actor
        self.HIDDEN_SIZE_2 = 64
        self.MAX_STEPS = 100

        self.RANGE_STEP_TO_WAIT = [3, 9]
        self.MIN_NUM_AGENT_IN_GROUP = 2


class REINFORCEHyperParams :
    def __init__(self):
        self.LR = 0.001
        self.BATCH_SIZE = 100
        self.GAMMA = 0.99

        self.HIDDEN_SIZE_1 = 128 #size of the first hidden layer of the actor
        self.HIDDEN_SIZE_2 = 64

        self.EPISODE_COUNT = 15000
        self.MAX_STEPS = 100

        self.RANGE_STEP_TO_WAIT = [3, 9]
        self.MIN_NUM_AGENT_IN_GROUP = 2


num_agents = 6
width=4
height=7

action_space = DiscreteActionSpace(5)
observation_space = DiscreteObservationSpace(4+5*num_agents)

