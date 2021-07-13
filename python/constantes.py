ACT_IN = 32
ACT_INTER = 64

CRIT_IN = 32
CRIT_INTER = 128

EPISODE_COUNT = 200

BUFFER_SIZE = 10**4  # replay buffer size
BATCH_SIZE = 256      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR_ACTOR = 0.0001     # learning rate of the actor 
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 0      # L2 weight decay

LEARNING_START = 25*BATCH_SIZE
MAX_STEPS = 1000