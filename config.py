from DQN.dqn import DQNAgent
from DQN.dueling_dqn import DuelingDQNAgent
# from SAC.sac import SACAgent

AGENT = {
    'DQN': DQNAgent,
    'DuelingDQN': DuelingDQNAgent,
    # 'SAC': SACAgent,
}