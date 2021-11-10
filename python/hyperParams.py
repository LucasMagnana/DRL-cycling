class HyperParams :
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


hyperParams = HyperParams()