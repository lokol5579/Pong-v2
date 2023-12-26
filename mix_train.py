import multiplayer
from collections import namedtuple, deque
import os
import numpy as np

class MixTrainConfig:
    def __init__(self):
        self.memo_1 = "MixTrain_DuelingDoubleDQN"
        self.memo_2_list = ["", "DQN", "DuelingDQN", "DuelingDoubleDQN"]
        self.agent_1 = "DuelingDoubleDQN"
        self.agent_2_list = ["", "DQN", "DuelingDQN", "DuelingDoubleDQN"]
        self.start_episode_1 = 0
        self.start_episode_2_list = [-1, 399, 384, 399]
        self.total_episode = 5
        self.horizon = 4
        self.player_list = [1, 2, 2, 2]
        self.skip_frame = 4
        self.test_mode = False
        self.mix_train = True
        self.mix_train_data = [0, -np.inf, -np.inf]
        self.memory = deque(maxlen=20000)
        self.config = {
            'model_dir': 'checkpoints',
            'video_dir': 'videos',
        }
        self.episode_per_model = 5

def mix_train(total_episode=100):
    steps_list, total_rews, eps_list = [], [], []

    config = MixTrainConfig()
    for j in range(total_episode):
        for i in range(len(config.memo_2_list)):
            config.memo_2 = config.memo_2_list[i]
            config.agent_2 = config.agent_2_list[i]
            config.start_episode_2 = config.start_episode_2_list[i]
            config.player = config.player_list[i]
            print()
            print("config.memo_1:", config.memo_1)
            print("config.agent_1:", config.agent_1)
            print("config.memo_2:", config.memo_2)
            print("config.agent_2:", config.agent_2)
            print("config.start_episode_2:", config.start_episode_2)
            print("config.total_episode:", config.total_episode)
            print("config.horizon:", config.horizon)
            print("config.player:", config.player)
            print("config.skip_frame:", config.skip_frame)
            print("config.test_mode:", config.test_mode)
            print("config.mix_train:", config.mix_train)
            print("config.mix_train_step:", config.mix_train_data)
            print("config.memory_len:", len(config.memory))
            print(config.config)
            print("config.episode_per_model:", config.episode_per_model)
            print()

            steps_list_, total_rews_, eps_list_, data_, memory_ = multiplayer.main(config)
            
            steps_list += steps_list_
            total_rews += total_rews_
            eps_list += eps_list_

            config.mix_train_data = data_
            config.memory = memory_
            config.start_episode_1 += config.episode_per_model
            config.total_episode = config.start_episode_1 + config.episode_per_model

    multiplayer.plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(config.config['model_dir'], config.memo_1, 'pong.png'))

if __name__ == '__main__':
    mix_train(total_episode=100)