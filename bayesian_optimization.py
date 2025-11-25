import json
import warnings
warnings.filterwarnings('ignore')
import gym
from collections import deque
import random
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

    # 超参数搜索空间
    def objective(self, trial: Trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        epsilon_start = trial.suggest_float("epsilon_start", 0.8, 1.0)
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
        epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 0.98, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000])
        update_frequency = trial.suggest_categorical("update_frequency", [1, 4, 10, 20])

        # 新增稳定性相关参数
        lr_decay = trial.suggest_float("lr_decay", 0.995, 0.9999)
        lr_decay_frequency = trial.suggest_categorical("lr_decay_frequency", [100, 200, 500])

        # 训练并评估（返回综合分数）
        composite_score = self.train_and_evaluate(
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_rate=epsilon_decay_rate,
            batch_size=batch_size, buffer_size=buffer_size,
            update_frequency=update_frequency,
            lr_decay=lr_decay,
            lr_decay_frequency=lr_decay_frequency,
            num_episodes=800  # 增加训练回合数以更好观察稳定性
        )

        return composite_score

    # 训练智能体并返回稳定性评估分数
    def train_and_evaluate(self, lr, gamma, epsilon_start, epsilon_end,
                           epsilon_decay_rate, batch_size, buffer_size,
                           update_frequency, lr_decay=0.999, lr_decay_frequency=200,
                           num_episodes=800, max_steps_per_episode=500):
        env = self.env

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

        # 训练监控
        episode_returns = []
        eval_returns = []
        stability_scores = []

        # 经验缓冲区
        buffer = deque(maxlen=buffer_size)
        update_count = 0
        learning_steps = 0

        for episode in range(num_episodes):
            state = env.reset()
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))
            episode_return = 0

            for step in range(max_steps_per_episode):
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                episode_return += reward

                # 控制学习频率
                update_count += 1
                if len(buffer) >= batch_size and update_count % update_frequency == 0:
                    batch = random.sample(buffer, batch_size)
                    agent.learn(batch, gamma)
                    learning_steps += 1

                    # 学习率衰减
                    if learning_steps % lr_decay_frequency == 0:
                        if hasattr(agent, 'update_learning_rate'):
                            agent.update_learning_rate(agent.lr * lr_decay)

                state = next_state
                if done:
                    break

            episode_returns.append(episode_return)

            # 更频繁的评估和稳定性检查
            if episode % 10 == 0:  # 每10回合评估一次
                current_eval = self.evaluate_policy(agent, num_episodes=3)
                eval_returns.append(current_eval)

                # 计算稳定性分数（最近几次评估的变异系数）
                if len(eval_returns) >= 5:
                    recent_returns = eval_returns[-5:]
                    stability = 1.0 - (np.std(recent_returns) / (np.mean(recent_returns) + 1e-8))
                    stability_scores.append(stability)

        # 最终综合评估
        final_performance = self.comprehensive_evaluation(
            agent, episode_returns, eval_returns, stability_scores
        )
        return final_performance

    # 综合评估性能、稳定性和收敛性
    def comprehensive_evaluation(self, agent, episode_returns, eval_returns, stability_scores):
        # 1. 最终性能（最后10次评估的平均）
        if len(eval_returns) >= 10:
            final_perf = np.mean(eval_returns[-10:])
        else:
            final_perf = np.mean(eval_returns) if eval_returns else 0

        # 2. 稳定性分数（避免剧烈波动）
        stability_weight = 0.3
        if stability_scores:
            stability = np.mean(stability_scores)
        else:
            stability = 0.5  # 默认稳定性

        # 3. 收敛性检查（后期是否保持稳定）
        convergence_weight = 0.2
        if len(eval_returns) >= 20:
            first_half = np.mean(eval_returns[:10])
            second_half = np.mean(eval_returns[-10:])
            # 如果后期性能没有下降，给予奖励
            convergence_bonus = 1.0 if second_half >= first_half * 0.9 else 0.0
        else:
            convergence_bonus = 0.0

        # 4. 峰值性能（考虑能达到的最高水平）
        peak_performance = max(eval_returns) if eval_returns else 0
        peak_weight = 0.2

        # 综合评分
        composite_score = (
                final_perf * 0.5 +  # 最终性能 50%
                stability * 100 * stability_weight +  # 稳定性 30%
                convergence_bonus * 100 * convergence_weight +  # 收敛性 20%
                peak_performance * peak_weight  # 峰值性能 20%
        )

        # 对不稳定性的惩罚
        if stability < 0.7:  # 如果稳定性较差
            composite_score *= 0.8

        # 对低性能的额外惩罚
        if final_perf < 100:
            composite_score *= 0.5

        return composite_score

    # 评估策略
    def evaluate_policy(self, agent, num_episodes=5):
        """更稳健的"""
        total_returns = []

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_return = 0

            while not done:
                action = agent.act(state, eps=0.0)  # 测试时完全贪婪
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_return += reward

                # 安全限制，防止无限循环
                if episode_return >= 500:  # CartPole的最大步数
                    break

            total_returns.append(episode_return)

        return np.mean(total_returns)

    # 执行优化并保存详细结果
    def optimize(self):
        study = create_study(
            direction="maximize",
            sampler=TPESampler(seed=1234)
        )

        print(f"开始稳定性优化的贝叶斯优化，共 {self.n_trials} 次试验...")
        study.optimize(self.objective, n_trials=self.n_trials)

        # 保存最佳参数
        self.best_params = study.best_params
        self.best_value = study.best_value

        print(f"\n优化完成！")
        print(f"最佳综合评分: {self.best_value:.2f}")
        print(f"最佳参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        # 保存详细结果
        self.save_detailed_results(study)

        return self.best_params

    # 将最佳参数保存为json格式，便于在main.py中调用
    def save_detailed_results(self, study):
        """保存详细的优化结果用于分析"""
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'all_trials': []
        }

        for trial in study.trials:
            results['all_trials'].append({
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            })

        output_file = f"best_params_{self.agent_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # 同时保存最佳参数到单独文件
        best_params_file = f"best_params_{self.agent_name}.json"
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)

        print(f"详细结果已保存到: {output_file}")
        print(f"最佳参数已保存到: {best_params_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="dqn", choices=["dqn", "ddqn"])
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    optimizer = BayesianOptimizer(agent_name=args.agent_name, n_trials=args.n_trials)
    best_params = optimizer.optimize()