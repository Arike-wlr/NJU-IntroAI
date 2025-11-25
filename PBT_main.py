import gym
from collections import deque
import random
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from agent import DQNAgent, DDQNAgent


class PBTTrainer:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.agents = []
        self.performance_history = []
        # 记录训练过程中的最大/最小/平均回报
        self.max_returns_history = []
        self.avg_returns_history = []
        self.min_returns_history = []

        # 初始化种群，每个智能体有随机超参数
        for i in range(population_size):
            agent_config = {
                # 学习相关参数
                'lr': 10 ** np.random.uniform(-5, -2),  # 10^-5 到 10^-2
                'gamma': np.random.uniform(0.9, 0.999),
                # 探索相关参数
                'epsilon_start': np.random.uniform(0.8, 1.0),
                'epsilon_end': np.random.uniform(0.01, 0.1),
                'epsilon_decay_rate': np.random.uniform(0.98, 0.999),
                # 经验回放参数
                'buffer_size': int(np.random.choice([5000, 10000, 20000, 50000])),
                'batch_size': 2 ** np.random.randint(5, 10),  # 32 到 512
                'update_frequency': int(np.random.choice([1, 4, 10, 20])),
                'agent_id': i
            }
            self.agents.append(agent_config)

    def exploit_and_explore(self, performances):
        """利用和探索阶段"""
        # 按性能排序
        sorted_indices = np.argsort(performances)[::-1]  # 降序

        for i in range(self.population_size):
            if i in sorted_indices[self.population_size // 2:]:  # 下半部分
                # 利用：从上半部分随机选择一个父代
                parent_idx = random.choice(sorted_indices[:self.population_size // 2])

                # 复制父代的参数
                self.agents[i] = self.agents[parent_idx].copy()

                # 探索：随机扰动所有参数
                # 探索策略参数
                self.agents[i]['epsilon_start'] = np.clip(
                    self.agents[i]['epsilon_start'] * np.random.uniform(0.9, 1.1),
                    0.5, 1.0
                )
                self.agents[i]['epsilon_end'] = np.clip(
                    self.agents[i]['epsilon_end'] * np.random.uniform(0.8, 1.2),
                    0.001, 0.2
                )
                self.agents[i]['epsilon_decay_rate'] = np.clip(
                    self.agents[i]['epsilon_decay_rate'] * np.random.uniform(0.98, 1.02),
                    0.95, 0.9999
                )

                # 学习参数
                self.agents[i]['gamma'] = np.clip(
                    self.agents[i]['gamma'] * np.random.uniform(0.99, 1.01),
                    0.9, 0.999
                )
                self.agents[i]['lr'] = np.clip(
                    self.agents[i]['lr'] * np.random.uniform(0.5, 2.0),
                    1e-6, 1e-2
                )

                # 经验回放参数
                self.agents[i]['buffer_size'] = int(np.clip(
                    self.agents[i]['buffer_size'] * np.random.uniform(0.7, 1.3),
                    512, 50000
                ))
                self.agents[i]['batch_size'] = int(np.clip(
                    self.agents[i]['batch_size'] * np.random.uniform(0.8, 1.2),
                    16, 1024
                ))
                self.agents[i]['update_frequency'] = int(np.clip(
                    self.agents[i]['update_frequency'] * np.random.uniform(0.8, 1.2),
                    1, 50
                ))

                print(f"Agent {i} 从 {parent_idx} 复制并探索新参数")

    def get_best_agent(self, performances):
        """获取性能最好的智能体配置"""
        best_idx = np.argmax(performances)
        return self.agents[best_idx], performances[best_idx]

    def print_population_status(self, performances, episode):
        """打印种群状态"""
        print(f"\n=== Episode {episode} 种群状态 ===")
        for i, (config, perf) in enumerate(zip(self.agents, performances)):
            print(f"Agent {i}: 回报={perf:.1f}, "
                  f"LR={config['lr']:.6f}, "
                  f"ε衰减={config['epsilon_decay_rate']:.4f}, "
                  f"批次={config['batch_size']}")
        best_config, best_perf = self.get_best_agent(performances)
        print(f"最佳智能体: ID={best_config['agent_id']}, 回报={best_perf:.1f}")

    def update_performance_history(self, performances, episode):
        """更新性能历史记录"""
        max_return = np.max(performances)
        avg_return = np.mean(performances)
        min_return = np.min(performances)

        self.max_returns_history.append((episode, max_return))
        self.avg_returns_history.append((episode, avg_return))
        self.min_returns_history.append((episode, min_return))

    def plot_performance(self, save_path=None):
        """绘制性能变化曲线"""
        if not self.max_returns_history:
            print("没有性能数据可绘制")
            return

        episodes = [item[0] for item in self.max_returns_history]
        max_returns = [item[1] for item in self.max_returns_history]
        avg_returns = [item[1] for item in self.avg_returns_history]
        min_returns = [item[1] for item in self.min_returns_history]

        plt.figure(figsize=(12, 8))

        # 绘制三条曲线
        plt.plot(episodes, max_returns, 'g-', linewidth=2, label='最大回报', alpha=0.8)
        plt.plot(episodes, avg_returns, 'b-', linewidth=2, label='平均回报', alpha=0.8)
        plt.plot(episodes, min_returns, 'r-', linewidth=2, label='最小回报', alpha=0.8)

        # 标记PBT更新点
        pbt_episodes = [ep for ep in episodes if ep % 10 == 0 and ep > 0]
        if pbt_episodes:
            pbt_returns = [max_returns[episodes.index(ep)] for ep in pbt_episodes]
            plt.scatter(pbt_episodes, pbt_returns, color='orange', s=50,
                        label='PBT更新点', zorder=5)

        plt.xlabel('Episode')
        plt.ylabel('回报')
        plt.title('PBT训练过程中种群回报变化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 设置y轴范围，让图像更美观
        if max_returns:
            plt.ylim(bottom=0, top=max(max_returns) * 1.1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能图表已保存到: {save_path}")

        plt.tight_layout()
        plt.show()

    def save_best_parameters(self, best_config, filename=None):
        """保存最佳参数到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_parameters_{timestamp}.json"

        parameters_to_save = {
            "lr": float(best_config["lr"]),
            "gamma": float(best_config["gamma"]),
            "epsilon_start": float(best_config["epsilon_start"]),
            "epsilon_end": float(best_config["epsilon_end"]),
            "epsilon_decay_rate": float(best_config["epsilon_decay_rate"]),
            "batch_size": int(best_config["batch_size"]),
            "buffer_size": int(best_config["buffer_size"]),
            "update_frequency": int(best_config["update_frequency"]),
            "final_max_return": float(self.max_returns_history[-1][1] if self.max_returns_history else 0)
        }

        with open(filename, 'w') as f:
            json.dump(parameters_to_save, f, indent=4)

        print(f"✅ 最佳参数已保存到: {filename}")
        return filename


def train_with_pbt(args):
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # 初始化PBT训练器
    pbt_trainer = PBTTrainer(population_size=10)

    # 为每个智能体创建对应的DQN agent和buffer
    agents = []
    buffers = []

    print("初始化PBT种群...")
    for config in pbt_trainer.agents:
        buffer = deque(maxlen=config['buffer_size'])
        if args.agent_name == "dqn":
            agent = DQNAgent(input_dim, output_dim, config['buffer_size'], 1234, config['lr'])
        else:
            agent = DDQNAgent(input_dim, output_dim, config['buffer_size'], 1234, config['lr'])

        agents.append(agent)
        buffers.append(buffer)
        print(f"Agent {config['agent_id']}: LR={config['lr']:.6f}, "
              f"Buffer={config['buffer_size']}, Batch={config['batch_size']}")

    # PBT训练循环
    for episode in range(args.num_episodes):
        performances = []

        # 并行训练所有智能体
        for i, (agent, buffer, config) in enumerate(zip(agents, buffers, pbt_trainer.agents)):
            epsilon = max(config['epsilon_end'], config['epsilon_start'] * (config['epsilon_decay_rate'] ** episode))

            # 训练一个episode
            state = env.reset()
            episode_return = 0
            losses = []

            for step in range(args.max_steps_per_episode):
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)

                buffer.append((state, action, reward, next_state, done))

                if len(buffer) >= config['batch_size'] and step % config['update_frequency'] == 0:
                    batch = random.sample(buffer, config['batch_size'])
                    loss = agent.learn(batch, config['gamma'])
                    if loss is not None:
                        losses.append(loss.item())

                episode_return += reward
                state = next_state
                if done:
                    break

            # 评估当前智能体的性能
            eval_return = eval_policy(agent, env)
            performances.append(eval_return)

        # 更新性能历史记录（每episode都记录）
        pbt_trainer.update_performance_history(performances, episode)

        # 每N个episode执行一次PBT的利用和探索
        pbt_interval = 10
        if episode % pbt_interval == 0 and episode > 0:
            print(f"\n=== 第{episode}episode执行PBT更新 ===")
            pbt_trainer.exploit_and_explore(performances)

            # 更新所有智能体的参数
            for i, config in enumerate(pbt_trainer.agents):
                for param_group in agents[i].optimizer.param_groups:
                    param_group['lr'] = config['lr']

                if buffers[i].maxlen != config['buffer_size']:
                    old_buffer = buffers[i]
                    buffers[i] = deque(maxlen=config['buffer_size'])
                    for experience in list(old_buffer)[-config['buffer_size']:]:
                        buffers[i].append(experience)

        # 每20个episode打印状态
        if episode % 20 == 0:
            pbt_trainer.print_population_status(performances, episode)

    # 训练结束
    best_config, best_performance = pbt_trainer.get_best_agent(performances)

    print(f"\n🎉 训练完成！最佳性能: {best_performance:.1f}")

    # 绘制性能图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"pbt_performance_{timestamp}.png"
    pbt_trainer.plot_performance(save_path=plot_filename)

    # 保存最佳参数
    saved_filename = pbt_trainer.save_best_parameters(best_config)

    env.close()
    return best_config, saved_filename, plot_filename


def eval_policy(agent, env, eval_episodes=3):
    """评估策略，取多次平均"""
    total_return = 0
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.act(state, eps=0.)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        total_return += episode_return
    return total_return / eval_episodes


def load_best_parameters(filename):
    """从JSON文件加载最佳参数"""
    with open(filename, 'r') as f:
        parameters = json.load(f)
    return parameters


if __name__ == "__main__":
    args = argparse.Namespace()
    args.agent_name = "dqn"
    args.num_episodes = 600
    args.max_steps_per_episode = 500

    best_config, params_filename, plot_filename = train_with_pbt(args)

    print(f"\n📊 性能图表: {plot_filename}")
    print(f"📂 参数文件: {params_filename}")

    # 演示如何加载保存的参数
    loaded_params = load_best_parameters(params_filename)
    print("\n加载的参数:")
    for key, value in loaded_params.items():
        print(f"  {key}: {value}")