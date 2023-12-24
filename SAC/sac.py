import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import namedtuple, deque
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义网络结构，输入为一张84*84的灰度图片，输出为各个动作的Q值，并采用2D卷积
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(state_size)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, action_size)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer = F.mish(self.conv1(state))
        layer = F.mish(self.conv2(layer))
        layer = F.mish(self.conv3(layer))
        # conv3 shape = Batch Size X n_filters X H X W
        layer = layer.view(layer.size()[0], -1)
        layer = F.mish(self.fc1(layer))
        layer = F.mish(self.fc2(layer))
        layer = self.fc3(layer)

        return layer
    
    def act(self, state, only_action=True):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if only_action:
            return action
        else:
            return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(state_size)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, action=None):
        layer = F.mish(self.conv1(state))
        layer = F.mish(self.conv2(layer))
        layer = F.mish(self.conv3(layer))
        # conv3 shape = Batch Size X n_filters X H X W
        layer = layer.view(layer.size()[0], -1)
        layer = F.mish(self.fc1(layer))
        layer = F.mish(self.fc2(layer))
        layer = self.fc3(layer)

        return layer


# 定义代理类
class SACAgent:
    def __init__(self, state_size, action_size, batch_size=64, lr=0.0001, memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr

        self.actor = Actor(state_size, action_size).to(device)
        self.critic_1 = Critic(state_size, action_size).to(device)
        self.critic_2 = Critic(state_size, action_size).to(device)
        self.target_critic_1 = Critic(state_size, action_size).to(device)
        self.target_critic_2 = Critic(state_size, action_size).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()))
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor(action_size)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha])

        # 创建记忆库
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state, eps, train=True):
        self.actor.eval()
        state = torch.from_numpy(np.float32(state)).unsqueeze(0).to(device)
        action = self.actor.act(state)
        return action.item()

    def memory_push(self, state, action, next_state, reward, done):
        # e = self.experience(state, action, next_state, reward, done)
        self.memory.append((state, action, next_state, reward, done))

    def memory_sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[i] for i in idxs])
        return (np.array(states), np.array(actions), np.array(next_states),
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))

    def update(self, step):
        if len(self.memory) < self.batch_size:
            return

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

        # 从记忆库中随机采样
        states, actions, next_states, rewards, dones = self.memory_sample(self.batch_size)

        states = torch.from_numpy(np.float32(states)).to(device)
        actions = torch.from_numpy(actions).to(device)
        next_states = torch.from_numpy(np.float32(next_states)).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        dones = torch.from_numpy(dones).to(device)


        with torch.no_grad():
            next_action, next_log_prob = self.actor.act(next_states, only_action=False)
            target_q1 = self.target_critic_1(next_states, next_action)
            target_q2 = self.target_critic_2(next_states, next_action)
            target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(1)
            target_q_value = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * 0.99 * target_q_min

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)

        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.act(states, only_action=False)
        q1_new = self.critic_1(states, new_action)
        q2_new = self.critic_2(states, new_action)
        q_new_min = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new_min).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    def save_model(self, episode, path):
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor_checkpoint_%d.pth' % episode))
        torch.save(self.critic_1.state_dict(), os.path.join(path, 'critic_1_checkpoint_%d.pth' % episode))
        torch.save(self.critic_2.state_dict(), os.path.join(path, 'critic_2_checkpoint_%d.pth' % episode))
        torch.save(self.target_critic_1.state_dict(), os.path.join(path, 'target_critic_1_checkpoint_%d.pth' % episode))
        torch.save(self.target_critic_2.state_dict(), os.path.join(path, 'target_critic_2_checkpoint_%d.pth' % episode))

    def load_model(self, episode, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor_checkpoint_%d.pth' % episode)))
        self.critic_1.load_state_dict(torch.load(os.path.join(path, 'critic_1_checkpoint_%d.pth' % episode)))
        self.critic_2.load_state_dict(torch.load(os.path.join(path, 'critic_2_checkpoint_%d.pth' % episode)))
        self.target_critic_1.load_state_dict(torch.load(os.path.join(path, 'target_critic_1_checkpoint_%d.pth' % episode)))
        self.target_critic_2.load_state_dict(torch.load(os.path.join(path, 'target_critic_2_checkpoint_%d.pth' % episode)))

    def update_epsilon(self, step):
        return 0.0