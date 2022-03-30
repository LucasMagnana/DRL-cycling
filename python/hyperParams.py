class DDQNHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 100000  #max size of the agent's replays buffer
        self.ALPHA = 0.05 #
        self.GAMMA = 0.9 #discount factor
        self.LR = 0.001 #learning rate of the actor
        self.BATCH_SIZE = 256 #number of replays processed by learning step

        self.HIDDEN_SIZE = 128 #size of the first hidden layer of the actor
        self.ACT_INTER = 64

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
        self.LR = 0.01
        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.LAMBDA = 0.99
        self.EPSILON = 0.2

        self.EPISODE_COUNT = 300
        self.NUM_AGENTS = 5
        self.NUM_EP_ENV = 1
        self.K = 3

        self.HIDDEN_SIZE = 32
        self.MAX_STEPS = 1000

        self.RANGE_STEP_TO_WAIT = [3, 9]
        self.MIN_NUM_AGENT_IN_GROUP = 2


module = "CartPole-v1" #"LunarLanderContinuous-v2"

