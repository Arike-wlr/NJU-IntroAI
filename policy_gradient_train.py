import os

os.environ['BOARD_SIZE'] = '5'
import json
import numpy as np
import tensorflow as tf
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent
from model_saver import ModelSaver
import time
import matplotlib.pyplot as plt


class TrainingMonitor:
    """训练监控器"""

    def __init__(self):
        self.deep_rewards = []
        self.rollout_rewards = []
        self.deep_win_rates = []
        self.rollout_win_rates = []
        self.start_time = time.time()

    def record_deep(self, episode, reward, win_rate, critic_loss=None, pi_loss=None):
        self.deep_rewards.append(reward)
        self.deep_win_rates.append(win_rate)

    def record_rollout(self, episode, reward, win_rate, critic_loss=None, pi_loss=None):
        self.rollout_rewards.append(reward)
        self.rollout_win_rates.append(win_rate)

    def plot_training(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 深层网络奖励
        if self.deep_rewards:
            axes[0, 0].plot(self.deep_rewards, 'b-', alpha=0.7, label='Deep')
            axes[0, 0].set_title('Deep_policy_reward')
            axes[0, 0].set_xlabel('evaluate intervals')
            axes[0, 0].set_ylabel('average')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # 浅层网络奖励
        if self.rollout_rewards:
            axes[0, 1].plot(self.rollout_rewards, 'r-', alpha=0.7, label='Rollout')
            axes[0, 1].set_title('Rollout_policy_reward')
            axes[0, 1].set_xlabel('evaluate intervals')
            axes[0, 1].set_ylabel('average')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # 深层网络胜率
        if self.deep_win_rates:
            axes[1, 0].plot(self.deep_win_rates, 'b-', alpha=0.7, label='Deep')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准')
            axes[1, 0].set_title('Deep_policy_win_rate')
            axes[1, 0].set_xlabel('evaluate intervals')
            axes[1, 0].set_ylabel('Win_rate')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # 浅层网络胜率
        if self.rollout_win_rates:
            axes[1, 1].plot(self.rollout_win_rates, 'r-', alpha=0.7, label='Rollout')
            axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%基准')
            axes[1, 1].set_title('Rollout_policy_win_rate')
            axes[1, 1].set_xlabel('evaluate intervals')
            axes[1, 1].set_ylabel('Win_rate')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('both_networks_training.png')
        plt.show()


def create_smart_opponent():
    """创建智能随机对手"""

    class SmartRandom:
        def __init__(self):
            self.board_size = 5
            # 定义位置价值：角>边>中心
            self.position_value = np.zeros((self.board_size, self.board_size))
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if (i == 0 and j == 0) or (i == 0 and j == self.board_size - 1) or \
                            (i == self.board_size - 1 and j == 0) or (
                            i == self.board_size - 1 and j == self.board_size - 1):
                        self.position_value[i, j] = 3.0  # 角
                    elif i == 0 or j == 0 or i == self.board_size - 1 or j == self.board_size - 1:
                        self.position_value[i, j] = 2.0  # 边
                    elif i == 2 and j == 2:
                        self.position_value[i, j] = 1.5  # 天元
                    else:
                        self.position_value[i, j] = 1.0  # 内部

        def select_action(self, position):
            legal_moves = position.all_legal_moves()
            legal_actions = np.where(legal_moves == 1)[0]

            if len(legal_actions) == 0:
                return 25  # PASS

            # 根据位置价值加权选择
            weights = []
            for action in legal_actions:
                if action == 25:
                    weights.append(0.1)  # PASS权重低
                else:
                    row, col = divmod(action, self.board_size)
                    weights.append(self.position_value[row, col])

            weights = np.array(weights)
            if weights.sum() == 0:
                return np.random.choice(legal_actions)

            weights = weights / weights.sum()
            return np.random.choice(legal_actions, p=weights)

    return SmartRandom()


def play_episode(agent, opponent_type="random", epsilon=0.1, is_evaluation=False):
    """玩一局游戏"""
    game = Position(komi=0.5)

    if opponent_type == "smart_random":
        opponent = create_smart_opponent()
    else:
        opponent = None

    while not game.is_game_over():
        current_player = 0 if game.to_play == 1 else 1

        if current_player == 0:  # 我们的AI
            # 训练时有探索，评估时没有
            if not is_evaluation and np.random.random() < epsilon:
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) == 0:
                    action = 25
                else:
                    action = np.random.choice(legal_actions)
            else:
                action, _ = agent.select_action(game, is_evaluation=is_evaluation)

            # 执行动作
            try:
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
            except Exception as e:
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) > 0:
                    action = legal_actions[0]
                    if action == 25:
                        game = game.pass_move(mutate=False)
                    else:
                        point = coords.from_flat(action)
                        game = game.play_move(point, mutate=False)

        else:  # 对手
            if opponent_type == "random":
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) == 0:
                    action = 25
                else:
                    action = np.random.choice(legal_actions)
            elif opponent_type == "smart_random":
                action = opponent.select_action(game)

            # 执行对手动作
            try:
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
            except Exception:
                pass

    # 计算结果
    result = game.result()
    reward = 1.0 if result > 0 else (-1.0 if result < 0 else 0.0)
    win = 1 if result > 0 else 0

    return reward, win, result


def evaluate_agent(agent, num_games=30, opponent_type="random"):
    """评估智能体"""
    wins = 0
    total_reward = 0

    for i in range(num_games):
        reward, win, _ = play_episode(
            agent,
            opponent_type=opponent_type,
            epsilon=0.0,  # 评估时不探索
            is_evaluation=True
        )
        wins += win
        total_reward += reward

    win_rate = wins / num_games
    avg_reward = total_reward / num_games

    return win_rate, avg_reward


def train_single_agent(agent, session, agent_type, hidden_layers, episodes,
                       model_saver, monitor, params=None):
    """训练单个智能体 - 修复版本"""
    print(f"\n=== 训练 {agent_type} 网络 ===")
    print(f"  网络结构: {hidden_layers}")
    print(f"  训练局数: {episodes}")

    # 如果传入了优化参数，应用到agent
    if params and agent_type == "deep":
        print(f"  使用优化参数:")
        print(f"    Critic学习率: {params.get('critic_lr', 0.01):.6f}")
        print(f"    Policy学习率: {params.get('pi_lr', 0.001):.6f}")
        print(f"    熵正则化: {params.get('entropy_cost', 0.01):.4f}")
        print(f"    批次大小: {params.get('batch_size', 32)}")
        print(f"    Critic更新次数: {params.get('num_critic_before_pi', 8)}")

        # 应用优化后的RL参数到已创建的agent
        agent.agent._critic_learning_rate = params.get('critic_lr', 0.01)
        agent.agent._pi_learning_rate = params.get('pi_lr', 0.001)
        agent.agent._entropy_cost = params.get('entropy_cost', 0.01)
        agent.agent._batch_size = params.get('batch_size', 32)
        agent.agent._num_critic_before_pi = params.get('num_critic_before_pi', 8)

    # 训练参数
    epsilon = 0.3
    epsilon_decay = 0.995
    epsilon_min = 0.05

    # 课程学习配置
    curriculum = [
        {"opponent": "random", "episodes": episodes // 3, "epsilon_decay": 0.99},
        {"opponent": "smart_random", "episodes": episodes // 2, "epsilon_decay": 0.995},
        {"opponent": "random", "episodes": episodes - (episodes // 3 + episodes // 2), "epsilon_decay": 0.997},
    ]

    total_episodes = 0
    eval_count = 0

    for stage_idx, stage in enumerate(curriculum):
        print(f"\n  阶段 {stage_idx + 1}/{len(curriculum)}: {stage['opponent']}对手")

        for stage_episode in range(stage['episodes']):
            # 训练一局
            play_episode(
                agent,
                opponent_type=stage['opponent'],
                epsilon=epsilon,
                is_evaluation=False
            )

            total_episodes += 1

            # 衰减探索率
            epsilon = max(epsilon_min, epsilon * stage['epsilon_decay'])

            # 定期评估
            if total_episodes % 50 == 0:
                eval_count += 1
                win_rate, avg_reward = evaluate_agent(
                    agent,
                    num_games=15,
                    opponent_type="random"
                )

                # 记录监控数据
                if agent_type == "deep":
                    monitor.record_deep(eval_count, avg_reward, win_rate)
                else:
                    monitor.record_rollout(eval_count, avg_reward, win_rate)

                print(f"    局数 {total_episodes}: 胜率={win_rate:.2%}, "
                      f"奖励={avg_reward:.3f}, 探索率={epsilon:.3f}")

            # 定期保存检查点
            if total_episodes % 100 == 0:
                model_name = f"{agent_type}_checkpoint_{total_episodes}"
                model_saver.save_policy_network(agent, model_name, iteration=total_episodes)
                print(f"    保存检查点: {model_name}")

    # 最终评估
    print(f"\n  {agent_type}网络最终评估:")
    for opponent in ["random", "smart_random"]:
        win_rate, avg_reward = evaluate_agent(agent, num_games=20, opponent_type=opponent)
        print(f"    对抗{opponent}: 胜率={win_rate:.2%}, 奖励={avg_reward:.3f}")

    return agent


def train_both_networks():
    """训练两个网络：深度网络和浅层网络"""
    print("A2C策略梯度训练 - 双网络版本")

    # 创建模型保存器
    model_saver = ModelSaver(save_dir="./saved_models")

    # 创建训练监控器
    monitor = TrainingMonitor()

    # 训练配置
    training_config = {
        "deep": {
            "agent_type": "deep",
            "hidden_layers": [256, 256],
            "episodes": 1200,
        },
        "rollout": {
            "agent_type": "rollout",
            "hidden_layers": [64],
            "episodes": 500,
        }
    }

    # 训练深度网络
    print("\n训练深度网络...")
    tf.reset_default_graph()
    deep_sess = tf.Session()
    deep_agent = GoPolicyAgent(
        session=deep_sess,
        hidden_layers=[256, 256],
        loss_str="a2c"
    )
    deep_sess.run(tf.global_variables_initializer())

    deep_agent = train_single_agent(
        agent=deep_agent,
        session=deep_sess,
        agent_type="deep",
        hidden_layers=[256, 256],
        episodes=1200,
        model_saver=model_saver,
        monitor=monitor,
        params=None  # 不使用优化参数
    )

    # 训练浅层网络
    print("\n训练浅层网络...")
    tf.reset_default_graph()
    rollout_sess = tf.Session()
    rollout_agent = GoPolicyAgent(
        session=rollout_sess,
        hidden_layers=[64],
        loss_str="a2c"
    )
    rollout_sess.run(tf.global_variables_initializer())

    rollout_agent = train_single_agent(
        agent=rollout_agent,
        session=rollout_sess,
        agent_type="rollout",
        hidden_layers=[64],
        episodes=500,
        model_saver=model_saver,
        monitor=monitor,
        params=None  # 不使用优化参数
    )

    print("保存最终模型")
    model_saver.save_policy_network(deep_agent, "deep_policy_final")
    model_saver.save_policy_network(rollout_agent, "rollout_policy_final")

    # 测试MCTS整合
    print("\n测试MCTS整合...")
    try:
        from algorimths.mcts import AlphaGoMCTS
        mcts = AlphaGoMCTS(
            deep_policy_agent="./saved_models/deep_policy_final",
            rollout_policy_agent="./saved_models/rollout_policy_final"
        )
        print("MCTS整合测试通过")
    except Exception as e:
        print(f"MCTS整合警告: {e}")

    # 列出保存的模型
    print("\n已保存的模型:")
    saved_models = model_saver.list_saved_models("policy")
    for model in saved_models:
        print(f"  - {model}")

    # 关闭所有会话
    deep_sess.close()
    rollout_sess.close()
    print("已关闭深度网络的会话")
    print("已关闭浅层网络的会话")

    # 绘制训练曲线
    print("生成训练曲线...")
    monitor.plot_training()
    print("训练完成！")

    return {"deep": deep_agent, "rollout": rollout_agent}


def train_both_networks_with_optimized_params():
    """使用优化后的参数训练两个网络 - 修复版本"""

    # 加载优化后的参数
    with open("./bayesian_optimization/best_params.json", 'r') as f:
        best_params = json.load(f)
    params = best_params['params']

    print("=" * 70)
    print("使用优化参数训练网络")
    print("=" * 70)
    print(f"MCTS模拟次数: {params.get('mcts_simulations', 100)}")
    print(f"MCTS探索权重: {params.get('exploration_weight', 1.0):.3f}")
    print(f"Rollout限制: {params.get('rollout_limit', 50)}")
    print(f"Critic学习率: {params.get('critic_lr', 0.01):.6f}")
    print(f"Policy学习率: {params.get('pi_lr', 0.001):.6f}")
    print(f"熵正则化: {params.get('entropy_cost', 0.01):.4f}")
    print(f"批次大小: {params.get('batch_size', 32)}")
    print(f"Critic更新次数: {params.get('num_critic_before_pi', 8)}")
    print("=" * 70)

    # 创建模型保存器
    model_saver = ModelSaver(save_dir="./saved_models_optimized")

    # 创建训练监控器
    monitor = TrainingMonitor()

    # 训练深度网络（使用优化参数）
    print("\n训练深度网络（使用优化参数）...")
    tf.reset_default_graph()
    deep_sess = tf.Session()
    deep_agent = GoPolicyAgent(
        session=deep_sess,
        hidden_layers=[256, 256],
        loss_str="a2c"
    )
    deep_sess.run(tf.global_variables_initializer())

    deep_agent = train_single_agent(
        agent=deep_agent,
        session=deep_sess,
        agent_type="deep",
        hidden_layers=[256, 256],
        episodes=1200,
        model_saver=model_saver,
        monitor=monitor,
        params=params  # 传递优化参数
    )

    # 训练浅层网络
    print("\n训练浅层网络...")
    tf.reset_default_graph()
    rollout_sess = tf.Session()
    rollout_agent = GoPolicyAgent(
        session=rollout_sess,
        hidden_layers=[64],
        loss_str="a2c"
    )
    rollout_sess.run(tf.global_variables_initializer())

    rollout_agent = train_single_agent(
        agent=rollout_agent,
        session=rollout_sess,
        agent_type="rollout",
        hidden_layers=[64],
        episodes=500,
        model_saver=model_saver,
        monitor=monitor,
        params=None  # 浅层网络不使用优化参数
    )

    print("保存最终模型")
    model_saver.save_policy_network(deep_agent, "deep_policy_optimized")
    model_saver.save_policy_network(rollout_agent, "rollout_policy_optimized")

    # 保存优化参数配置
    config_file = "./optimized_training_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            'mcts_params': {
                'simulations': params.get('mcts_simulations', 100),
                'exploration_weight': params.get('exploration_weight', 1.0),
                'rollout_limit': params.get('rollout_limit', 50)
            },
            'rl_params': {
                'critic_lr': params.get('critic_lr', 0.01),
                'pi_lr': params.get('pi_lr', 0.001),
                'entropy_cost': params.get('entropy_cost', 0.01),
                'batch_size': params.get('batch_size', 32),
                'num_critic_before_pi': params.get('num_critic_before_pi', 8)
            },
            'optimization_info': best_params
        }, f, indent=2)

    print(f"\n优化参数配置已保存到: {config_file}")

    # 测试MCTS整合（使用优化参数）
    print("\n测试优化MCTS整合...")
    try:
        from algorimths.mcts import AlphaGoMCTS
        mcts = AlphaGoMCTS(
            deep_policy_agent="./saved_models_optimized/deep_policy_optimized",
            rollout_policy_agent="./saved_models_optimized/rollout_policy_optimized"
        )
        print("优化MCTS整合测试通过")
    except Exception as e:
        print(f"优化MCTS整合警告: {e}")

    # 关闭会话
    deep_sess.close()
    rollout_sess.close()
    print("已关闭深度网络的会话")
    print("已关闭浅层网络的会话")

    # 绘制训练曲线
    print("生成训练曲线...")
    monitor.plot_training()
    print("优化训练完成！")

    return {"deep": deep_agent, "rollout": rollout_agent}


if __name__ == "__main__":
    # 选择训练模式
    import argparse

    parser = argparse.ArgumentParser(description="Mini AlphaGo训练")
    parser.add_argument("--mode", choices=["original", "optimized"], default="optimized",
                        help="训练模式: original-原始参数, optimized-优化参数")

    args = parser.parse_args()

    if args.mode == "original":
        print("使用原始参数训练...")
        trained_agents = train_both_networks()
    else:
        print("使用优化参数训练...")
        trained_agents = train_both_networks_with_optimized_params()