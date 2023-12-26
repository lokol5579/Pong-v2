from DQN.dqn import DQNAgent
from DQN.dueling_dqn import DuelingDQNAgent
from DQN.dueling_double_dqn import DuelingDoubleDQNAgent
from DQN.double_dqn import DoubleDQNAgent
# from SAC.sac import SACAgent

AGENT = {
    'DQN': DQNAgent,
    'DuelingDQN': DuelingDQNAgent,
    'DuelingDoubleDQN': DuelingDoubleDQNAgent,
    'DoubleDQN': DoubleDQNAgent,
    # 'SAC': SACAgent,
}