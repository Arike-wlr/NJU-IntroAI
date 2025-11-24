import json
import warnings

warnings.filterwarnings('ignore')
import gym
from collections import deque
import random
import torch
import numpy as np
from optuna import create_study, Trial
from optuna.samplers import TPESampler

from agent import DQNAgent, DDQNAgent


class BayesianOptimizer:
    def __init__(self, agent_name="dqn", n_trials=50):
        self.agent_name = agent_name
        self.n_trials = n_trials
        self.env = gym.make("CartPole-v1")
        self.best_params = None

    def objective(self, trial: Trial):
        # 完整的超参数搜索空间
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        epsilon_start = trial.suggest_float("epsilon_start", 0.8, 1.0)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 0.98, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000])
        update_frequency = trial.suggest_categorical("update_frequency", [1, 4, 10, 20, 50])

        # 固定参数
        num_episodes =600
        max_steps_per_episode = 500

        # 训练并评估
        total_return = self.train_and_evaluate(
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_rate=epsilon_decay_rate,
            batch_size=batch_size, buffer_size=buffer_size,
            update_frequency=update_frequency,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode
        )

        return total_return

    def train_and_evaluate(self, lr, gamma, epsilon_start, epsilon_end,
                           epsilon_decay_rate, batch_size, buffer_size,
                           update_frequency, hidden_layers=None,
                           num_episodes=300, max_steps_per_episode=500):
        """训练智能体并返回最终性能"""
        env = self.env
        buffer = deque(maxlen=buffer_size)

        # 初始化智能体
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        if self.agent_name == "dqn":
            agent = DQNAgent(
                input_dim, output_dim,
                buffer_size=buffer_size,
                seed=1234,
                lr=lr,
            )
        elif self.agent_name == "ddqn":
            agent = DDQNAgent(
                input_dim, output_dim,
                seed=1234,
                lr=lr,
            )

        # 训练循环
        best_eval_return = -float('inf')
        eval_interval = 20
        update_count = 0

        for episode in range(num_episodes):
            state = env.reset()
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

            for step in range(max_steps_per_episode):
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))

                # 使用update_frequency控制学习频率
                update_count += 1
                if len(buffer) >= batch_size and update_count % update_frequency == 0:
                    batch = random.sample(buffer, batch_size)
                    agent.learn(batch, gamma)

                state = next_state
                if done:
                    break

            # 定期评估
            if episode % eval_interval == 0:
                eval_return = self.evaluate_policy(agent, num_episodes=3)
                best_eval_return = max(best_eval_return, eval_return)

        # 最终评估
        final_return = self.evaluate_policy(agent, num_episodes=5)
        return final_return

    def evaluate_policy(self, agent, num_episodes=5):
        """评估策略，运行多次取平均"""
        total_returns = []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0
            while not done:
                action = agent.act(state, eps=0.)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_return += reward
            total_returns.append(episode_return)

        return np.mean(total_returns)

    def optimize(self):
        """执行贝叶斯优化"""
        study = create_study(
            direction="maximize",
            sampler=TPESampler(seed=1234)
        )

        print(f"开始贝叶斯优化，共 {self.n_trials} 次试验...")
        study.optimize(self.objective, n_trials=self.n_trials)

        # 保存最佳参数
        self.best_params = study.best_params
        self.best_value = study.best_value

        print(f"\n优化完成！")
        print(f"最佳回报: {self.best_value:.2f}")
        print(f"最佳参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        # 保存到JSON文件
        output_file = f"best_params_{self.agent_name}.json"
        with open(output_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)

        print(f"参数已保存到: {output_file}")
        return self.best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="dqn", choices=["dqn", "ddqn"])
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    optimizer = BayesianOptimizer(agent_name=args.agent_name, n_trials=args.n_trials)
    best_params = optimizer.optimize()