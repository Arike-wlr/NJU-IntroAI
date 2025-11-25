import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from q_network import QNetwork

# Define the DQN agent class
class DQNAgent:
    # Initialize the DQN agent
    def __init__(self, state_dim, action_dim, buffer_size, seed, lr, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)
        self.device = device

        self.qnetwork_local = QNetwork(state_dim, action_dim, seed).to(device)
        self.qnetwork_target = QNetwork(state_dim, action_dim, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr) #使用lr

        self.t_step = 0

    # Choose an action based on the current state
    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # 将numpy数组的state转换为PyTorch tensor，并添加batch维度
        self.qnetwork_local.eval()
        # 将网络设置为评估模式（关闭dropout等训练专用层）
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        # 禁用梯度计算，提高效率，使用本地Q网络计算当前状态下所有动作的Q值
        self.qnetwork_local.train()
        # 恢复训练模式

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.action_dim)
        # ε-贪婪策略：以eps概率随机探索，以(1-eps)概率选择最优动作

    def act_no_explore(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        # 关键区别：总是选择Q值最大的动作，不进行随机探索
        return action_values.argmax(dim=1).item()

    # Learn from batch of experiences
    def learn(self, experiences, gamma):
        device = self.device
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # 这里用到了gamma
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
        return loss

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Define the DDQN agent class
class DDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, buffer_size,seed, lr, device="cpu"):
        super().__init__(state_dim, action_dim, buffer_size,seed, lr, device)


    # Learn from batch of experiences
    def learn(self, experiences, gamma):
        device = self.device
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        action_max = torch.argmax(self.qnetwork_local(next_states), dim=1).unsqueeze(1)
        Q_targets = rewards + (gamma * torch.gather(self.qnetwork_target(next_states).detach(), 1, action_max) * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
        return loss
