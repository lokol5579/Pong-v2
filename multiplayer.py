import retro
import matplotlib.pyplot as plt
import gym
from IPython import display
import gym.spaces
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os

from config import *

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from gym import make, ObservationWrapper, Wrapper
from gym.spaces import Box
from collections import deque

import signal

CONFIG = {
    'model_dir': 'checkpoints',
    'video_dir': 'videos',
}

total_rews = []
steps_list = []
eps_list = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_mode", action="store_true", default=False)

    parser.add_argument("-memo_1", type=str, default='test')
    parser.add_argument("-memo_2", type=str, default='test')
    parser.add_argument("-agent_1", type=str, default='DQN')
    parser.add_argument("-agent_2", type=str, default='DQN')
    parser.add_argument("-start_episode_1", type=int, default=0)
    parser.add_argument("-start_episode_2", type=int, default=0)

    parser.add_argument("-total_episode", type=int, default=400)
    parser.add_argument("-horizon", type=int, default=4)
    parser.add_argument("-player", type=int, default=1)
    parser.add_argument("-skip_frame", type=int, default=4)
    return parser.parse_args()


class FrameSampleTool(ObservationWrapper):
    def __init__(self, env):
        super(FrameSampleTool, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self._width = 84
        self._height = 84

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

    def step(self, act1, act2):
        obs, rew, done, info = self.env.step(act1, act2)
        obs = self.observation(obs)
        return obs, rew, done, info

class NSkipTool(Wrapper):
    def __init__(self, env, skip_frame=4, players=1):
        super(NSkipTool, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self.skip_frame = skip_frame
        self.players = players

    def step(self, act1, act2):
        total_reward = 0.0
        done = None
        for _ in range(self.skip_frame):
            obs, reward, done, log = self.env.step(act1, act2)
            self.obs_buffer.append(obs)
            if self.players == 2:
                reward = reward[0]
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frame, total_reward, done, log

    def reset(self):
        self.obs_buffer.clear()
        obs = self.env.reset()
        self.obs_buffer.append(obs)
        return obs



class FrameBufferTool(ObservationWrapper):
    def __init__(self, env, horizon, dtype=np.float32):
        super(FrameBufferTool, self).__init__(env)
        obs_space = env.observation_space
        self._dtype = dtype
        self.observation_space = Box(obs_space.low.repeat(horizon, axis=0),
                                     obs_space.high.repeat(horizon, axis=0), dtype=self._dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self._dtype)
        return self.observation(self.env.reset())

    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer

    def step(self, act1, act2):
        obs, rew, done, info = self.env.step(act1, act2)
        obs = self.observation(obs)
        return obs, rew, done, info

class ReplaceTool(ObservationWrapper):
    def __init__(self, env):
        super(ReplaceTool, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(obs_shape[::-1]), dtype=np.float32)

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)
    
    def step(self, act1, act2):
        obs, rew, done, info = self.env.step(act1, act2)
        obs = self.observation(obs)
        return obs, rew, done, info

class NormTool(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

    def step(self, act1, act2):
        obs, rew, done, info = self.env.step(act1, act2)
        obs = self.observation(obs)
        return obs, rew, done, info

def wrap_env(skip_frame=4, players=1, horizon=4):
    env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=players), players=players)
    env = NSkipTool(env, skip_frame, players=players)
    env = FrameSampleTool(env)
    env = ReplaceTool(env)
    env = FrameBufferTool(env, horizon)
    env = NormTool(env)
    return env


def PongDiscretizer(env, players=1):
    """
    Discretize Retro Pong-Atari2600 environment
    """
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=[['DOWN'], ['UP'], ['BUTTON'],], players=players)
    

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos, players=1):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.players = players
        self._decode_discrete_action = []
        self._decode_discrete_action2 = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        if self.players == 2:
            # pkayer 2 : 7: DOWN, 6: 'UP', 15:'BUTTOM'
            arr = np.array([False] * env.action_space.n)
            arr[7] = True
            self._decode_discrete_action2.append(arr)
            
            arr = np.array([False] * env.action_space.n)
            arr[6] = True
            self._decode_discrete_action2.append(arr)
            
            arr = np.array([False] * env.action_space.n)
            arr[15] = True
            self._decode_discrete_action2.append(arr)
        
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act1, act2):
        act1_v = self._decode_discrete_action[act1].copy()
        if self.players == 1:
            return act1_v.copy()
        else:
            act2_v = self._decode_discrete_action2[act2].copy()
            return np.logical_or(act1_v, act2_v).copy()
    
    def step(self, act1, act2=None):
        return self.env.step(self.action(act1, act2))


def traverse_imgs(writer, images):
    # 遍历所有图片，并且让writer抓取视频帧
    with tqdm(total=len(images), desc='traverse_imgs', leave=False) as pbar:
        for img in images:
            plt.imshow(img)
            writer.grab_frame()
            plt.pause(0.01)
            plt.clf()
            pbar.update(1)
        plt.close()


def plot_learning_curve(x, scores, epsilon, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilon, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x,running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    plt.savefig(filename)


def train(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, total_episode=1000):
    env = wrap_env(skip_frame=skip_frame, players=players, horizon=horizon)
    env.reset()

    global total_rews
    global steps_list
    global eps_list
    best_avg_rew = -np.inf
    best_rew = -np.inf
    steps = 0

    for i in range(total_episode):
        done = False
        total_rew = 0.0
        obs = env.reset()

        while not done:
            # 更新epsilon
            eps = agent_1.update_epsilon(steps)
                
            # 右侧板
            action_1 = agent_1.select_action(obs, eps)

            # 左侧板
            if players == 2 and agent_2 is None:
                raise ValueError("agent_2 is None")
            if players == 2 and agent_2 is not None:
                action_2 = agent_2.select_action(obs[:, :, ::-1].copy(), eps=0.0)
            else:
                action_2 = None

            nxt_obs, rew, done, info = env.step(action_1, action_2)
            agent_1.memory_push(obs, action_1, nxt_obs, rew, done)
            
            obs = nxt_obs
            total_rew += rew
            steps += 1

            agent_1.update(steps)

        total_rews.append(total_rew)
        steps_list.append(steps)
        eps_list.append(eps)

        if total_rew > best_rew:
            best_rew = total_rew

        avg_total_rew = np.mean(total_rews[-100:])

        if avg_total_rew > best_avg_rew:
            best_avg_rew = avg_total_rew
            agent_1.save_model(i, CONFIG['model_dir'])
            if args.player == 2:
                agent_2.save_model(i, CONFIG['model_dir'])

        print('episode: %d, total step = %d, total reward = %.2f, avg reward = %.6f, best reward = %.2f, best avg reward = %.6f, epsilon = %.6f' % (i, steps, total_rew, avg_total_rew, best_rew, best_avg_rew, eps))

        if i % 25 == 0:
            # 测试agent
            test(agent_1, agent_2, players=players, skip_frame=skip_frame, horizon=horizon, max_steps=max_steps, episode=i, env=env)

    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))

def test(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, episode=0, env=None):
    if env is None:
        env = wrap_env(skip_frame=skip_frame, players=players, horizon=horizon)
        env.reset()

    done = False
    steps = 0
    images = []
    obs = env.reset()

    while not done:
        # 右侧板
        action_1 = agent_1.select_action(obs, eps=0.0)

        # 左侧板
        if players == 2 and agent_2 is None:
            raise ValueError("agent_2 is None")
        if players == 2 and agent_2 is not None:
            action_2 = agent_2.select_action(obs[:, :, ::-1].copy(), eps=0.0)
        else:
            action_2 = None

        nxt_obs, rew, done, info = env.step(action_1, action_2)

        obs = nxt_obs
        steps += 1

        if steps % 8 == 0:
            images.append(env.render(mode='rgb_array'))

        if steps > max_steps:
            break


    # 创建video writer, 设置好相应参数，fps
    metadata = dict(title='01', artist='Matplotlib',comment='depth prediiton')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    figure = plt.figure(figsize=(10.8, 7.2))
    plt.ion()                                   # 为了可以动态显示
    plt.tight_layout()                          # 尽量减少窗口的留白
    with writer.saving(figure, os.path.join(CONFIG['video_dir'], 'episode_%d.mp4' % episode), 100): 
        traverse_imgs(writer, images)

    return info

def main(args):
    # 检查目录是否存在，如果不存在则创建，存在则停止运行，test_mode不需要创建

    if args.player == 2 and not os.path.exists(os.path.join(CONFIG['model_dir'], args.memo_1)):
        raise ValueError("agent_2 model is not exists")

    if args.player == 1:
        CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1)
        CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1)

    if not args.test_mode and args.memo_1 != 'test' and args.player != 2 and os.path.exists(CONFIG['model_dir']):
        raise ValueError("memo is already exists")

    if args.player == 1:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3)
        agent_2 = None
    elif args.player == 2:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3)
        agent_2 = AGENT[args.agent_2](state_size=(args.horizon, 84, 84), action_size=3)

    if args.test_mode:
        if args.player == 2:
            agent_1.load_model(args.start_episode_1, os.path.join(CONFIG['model_dir'], args.memo_1))
            agent_2.load_model(args.start_episode_2, os.path.join(CONFIG['model_dir'], args.memo_2))

            CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1 + '_' + args.memo_2)
            CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1 + '_' + args.memo_2)

            if not os.path.exists(CONFIG['model_dir']):
                os.makedirs(CONFIG['model_dir'])
            if not os.path.exists(CONFIG['video_dir']):
                os.makedirs(CONFIG['video_dir'])
        else:
            if not os.path.exists(CONFIG['model_dir']):
                raise ValueError("model dir is not exists")
            if not os.path.exists(CONFIG['video_dir']):
                raise ValueError("video dir is not exists")
            
            agent_1.load_model(args.start_episode_1, CONFIG['model_dir'])
        
        # 测试agent
        info = test(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, episode=args.start_episode_1)
        print(info)
    else:
        if args.player == 2:
            agent_1.load_model(args.start_episode_1, os.path.join(CONFIG['model_dir'], args.memo_1))
            agent_2.load_model(args.start_episode_2, os.path.join(CONFIG['model_dir'], args.memo_2))

            CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1 + '_' + args.memo_2)
            CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1 + '_' + args.memo_2)

            if not os.path.exists(CONFIG['model_dir']):
                os.makedirs(CONFIG['model_dir'])
            else:
                raise ValueError("model dir is already exists")
            if not os.path.exists(CONFIG['video_dir']):
                os.makedirs(CONFIG['video_dir'])
            else:
                raise ValueError("video dir is already exists")
        else:
            if args.memo_1 != 'test':
                if os.path.exists(CONFIG['model_dir']):
                    raise ValueError("model dir is already exists")
                else:
                    os.makedirs(CONFIG['model_dir'])

                if os.path.exists(CONFIG['video_dir']):
                    raise ValueError("video dir is already exists")
                else:
                    os.makedirs(CONFIG['video_dir'])

        # 训练agent
        train(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, total_episode=args.total_episode)


def int_handler(signum, frame):
    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, int_handler)

    args = parse_args()    
    main(args)