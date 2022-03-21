class ContHyperParams :
    def __init__(self):
        self.ACT_IN = 400
        self.ACT_INTER = 300

        self.CRIT_IN = 400
        self.CRIT_INTER = 300

        self.EPISODE_COUNT = 5000
        self.POLICY_DELAY = 2

        self.BUFFER_SIZE = 5e5  # replay buffer size
        self.BATCH_SIZE = 100      # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 5e-3           # for soft update of target parameters
        self.LR_ACTOR = 0.001     # learning rate of the actor 
        self.LR_CRITIC = 0.001       # learning rate of the critic
        self.WEIGHT_DECAY = 0      # L2 weight decay

        self.POLICY_NOISE = 0.2
        self.NOISE_CLIP = 0.5

        self.LEARNING_START = 25*self.BATCH_SIZE
        self.MAX_STEPS = 2000
        self.EXPLORATION_NOISE = 0.1


class DisHyperParams :
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


#Cartpole hyper parameters (solves it but not optimal)
'''class DisHyperParams :
    def __init__(self):
        self.BUFFER_SIZE = 1e25 
        self.ALPHA = 0.05 #
        self.GAMMA = 0.9
        self.LR = 0.001
        self.BATCH_SIZE = 32

        self.HIDDEN_SIZE = 32
        self.ACT_INTER = 32

        self.EPISODE_COUNT = 3000
        self.MAX_STEPS = 1000
        self.LEARNING_START = 0

        self.EPSILON = 1.0
        self.MIN_EPSILON = 0
        self.EPSILON_DECAY = self.EPSILON/(self.EPISODE_COUNT*4/5)

        #specific to custom env

        self.NUM_RNN_LAYERS = 1
        self.SEQ_SIZE = 32'''


module = "monresovelo" #"CartPole-v1"

if("Continuous" in module):
    hyperParams = ContHyperParams()
else:
    hyperParams = DisHyperParams()