ACT_IN = 400
ACT_INTER = 300

CRIT_IN = 400
CRIT_INTER = 300

EPISODE_COUNT = 10000
POLICY_DELAY = 2

BUFFER_SIZE = 5e5  # replay buffer size
BATCH_SIZE = 100      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3           # for soft update of target parameters
LR_ACTOR = 0.001     # learning rate of the actor 
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 0      # L2 weight decay

POLICY_NOISE = 0.2
NOISE_CLIP = 0.5

LEARNING_START = 25*BATCH_SIZE
MAX_STEPS = 2000
EXPLORATION_NOISE = 0.1