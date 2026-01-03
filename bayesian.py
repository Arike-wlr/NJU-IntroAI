import os
os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import json
import time
from datetime import datetime
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent
from opponent_pool import OpponentPool
from .algorimths.mcts import AlphaGoMCTS


class AlphaGoBayesianOptimizer:
    """AlphaGo贝叶斯优化器"""

    def __init__(self,
                 num_eval_games=20,
                 log_dir="./bayesian_opt_logs"):

        self.num_eval_games = num_eval_games
        self.log_dir = log_dir
        self.best_params = None
        self.best_score = -float('inf')
        os.makedirs(log_dir, exist_ok=True)
        self.pbounds = {
            'mcts_simulations': (10, 200),  # MCTS模拟次数
            'exploration_weight': (0.1, 2.0),  # 探索权重
            'critic_lr': (1e-5, 1e-2),  # Critic学习率
            'pi_lr': (1e-6, 1e-3),  # Pi学习率
            'entropy_cost': (0.0, 0.1),  # 熵系数
            'batch_size_log': (4, 7),  # 批次大小对数(log2)
            'num_critic_before_pi': (1, 16),  # Critic更新频率
            'rollout_limit': (10, 100),  # rollout步数限制
        }
        self.optimizer = BayesianOptimization(
            f=self.evaluate_parameters,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = JSONLogger(path=f"{log_dir}/optimization_{timestamp}.json")
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)
        self.results_file = f"{log_dir}/results_{timestamp}.txt"

    def convert_parameters(self, params):
        """转换参数到合适范围"""
        converted = params.copy()

        # 转换批次大小：2^log_value
        converted['batch_size'] = int(2 ** params['batch_size_log'])

        # 转换学习率为对数空间
        converted['critic_lr'] = 10 ** (-np.random.uniform(2, 5))
        converted['pi_lr'] = converted['critic_lr'] * 0.1

        # 确保整数参数
        converted['mcts_simulations'] = int(params['mcts_simulations'])
        converted['num_critic_before_pi'] = int(params['num_critic_before_pi'])
        converted['rollout_limit'] = int(params['rollout_limit'])

        return converted

    def create_agent_with_params(self, params, agent_type="deep"):
        """使用给定参数创建agent"""

        hidden_layers = [256, 256] if agent_type == "deep" else [64]

        # 创建会话和agent
        tf.reset_default_graph()
        session = tf.Session()

        agent = GoPolicyAgent(
            session=session,
            hidden_layers=hidden_layers,
            loss_str="a2c"
        )

        # 设置超参数到agent（通过修改agent.agent的属性）
        agent.agent._critic_learning_rate = params['critic_lr']
        agent.agent._pi_learning_rate = params['pi_lr']
        agent.agent._entropy_cost = params['entropy_cost']
        agent.agent._batch_size = params['batch_size']
        agent.agent._num_critic_before_pi = params['num_critic_before_pi']

        session.run(tf.global_variables_initializer())

        return agent, session

    def create_mcts_with_params(self, deep_agent, rollout_agent, params):
        """使用给定参数创建MCTS"""
        mcts = AlphaGoMCTS(
            deep_policy_agent=None,
            rollout_policy_agent=None,
            exploration_weight=params['exploration_weight'],
            simulation_limit=params['rollout_limit']
        )

        # 直接设置网络
        mcts.deep_policy = deep_agent
        mcts.rollout_policy = rollout_agent

        return mcts

    def play_evaluation_game(self, agent, mcts=None, opponent="random", params=None):
        """玩一局评估游戏"""
        game = Position(komi=0.5)

        while not game.is_game_over():
            current_player = 0 if game.to_play == 1 else 1

            if current_player == 0:  # 我们的AI
                if mcts and params:
                    action, _ = mcts.get_best_action(
                        game,
                        current_player,
                        num_simulations=int(params['mcts_simulations'])
                    )
                else:
                    action, _ = agent.select_action(game, is_evaluation=True)
            else:  # 对手
                if opponent == "random":
                    legal_moves = game.all_legal_moves()
                    legal_actions = np.where(legal_moves == 1)[0]
                    if len(legal_actions) == 0:
                        action = 25
                    else:
                        action = np.random.choice(legal_actions)
                elif isinstance(opponent, GoPolicyAgent):
                    action, _ = opponent.select_action(game, is_evaluation=True)

            # 执行动作
            try:
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
            except Exception:
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) > 0:
                    action = legal_actions[0]
                    if action == 25:
                        game = game.pass_move(mutate=False)
                    else:
                        point = coords.from_flat(action)
                        game = game.play_move(point, mutate=False)

        return game.result()

    def quick_train_and_evaluate(self, params, num_train_games=30):
        """快速训练和评估"""

        # 创建深度和浅层网络
        deep_agent, deep_sess = self.create_agent_with_params(params, "deep")
        rollout_agent, rollout_sess = self.create_agent_with_params(params, "rollout")

        # 创建MCTS
        mcts = self.create_mcts_with_params(deep_agent, rollout_agent, params)

        # 快速训练（少量游戏）
        print(f"\n🚀 快速训练 {num_train_games} 局...")
        train_results = []

        for game_idx in range(num_train_games):
            result = self.play_evaluation_game(
                deep_agent,
                mcts=None,  # 训练时不使用MCTS
                opponent="random",
                params=params
            )
            train_results.append(result)

            # 模拟训练更新（实际需要调用agent的训练方法）
            # 这里简化：只记录结果

        # 评估（使用MCTS）
        print(f"📊 评估 {self.num_eval_games} 局...")
        eval_results = []

        for i in range(self.num_eval_games):
            result = self.play_evaluation_game(
                deep_agent,
                mcts=mcts,
                opponent="random"
            )
            eval_results.append(result)

            if (i + 1) % 5 == 0:
                print(f"  评估完成 {i + 1}/{self.num_eval_games} 局")

        # 计算得分
        win_rate = np.mean([1 if r > 0 else 0 for r in eval_results])
        avg_result = np.mean(eval_results)
        score = win_rate * 100 + avg_result * 10
        penalty = 0
        if params['critic_lr'] > 0.01:
            penalty += 5
        if params['entropy_cost'] < 0.001:
            penalty += 3

        final_score = score - penalty

        # 清理
        deep_sess.close()
        rollout_sess.close()
        tf.reset_default_graph()

        return final_score

    def evaluate_parameters(self, **kwargs):
        """贝叶斯优化的目标函数"""
        params = self.convert_parameters(kwargs)

        print(f"\n{'=' * 60}")
        print(f"🔬 测试参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        try:
            score = self.quick_train_and_evaluate(params)
            print(f"📈 得分: {score:.4f}")
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

                # 保存最佳参数
                with open(f"{self.log_dir}/best_params.json", "w") as f:
                    json.dump({
                        'score': float(self.best_score),
                        'params': self.best_params,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                print(f"🎉 新的最佳得分: {score:.4f}")

            # 记录到结果文件
            with open(self.results_file, "a") as f:
                f.write(f"{datetime.now().isoformat()}\n")
                f.write(f"得分: {score:.4f}\n")
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")
                f.write("-" * 40 + "\n")

            return score

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return -100  # 返回极低分

    def optimize(self, n_iter=50, init_points=10):
        """运行贝叶斯优化"""
        print("🎯 开始贝叶斯优化调参")
        print(f"迭代次数: {n_iter}")
        print(f"初始点: {init_points}")
        print(f"评估游戏数: {self.num_eval_games}")
        start_time = time.time()

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 打印最佳结果
        print("\n" + "=" * 60)
        print("🏆 优化完成!")
        print(f"总时间: {elapsed_time:.1f}秒")
        print(f"平均每轮: {elapsed_time / (init_points + n_iter):.1f}秒")
        print("=" * 60)

        print(f"\n🎯 最佳得分: {self.best_score:.4f}")
        print("🎯 最佳参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        # 保存最终报告
        self.save_final_report(elapsed_time)

        return self.optimizer.max

    def save_final_report(self, elapsed_time):
        """保存最终报告"""
        report_file = f"{self.log_dir}/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Mini AlphaGo 贝叶斯优化调参报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {elapsed_time:.1f}秒\n\n")

            f.write("🏆 最佳参数配置:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n最佳得分: {self.best_score:.4f}\n\n")

            f.write("📊 搜索参数范围:\n")
            f.write("-" * 40 + "\n")
            for key, (low, high) in self.pbounds.items():
                f.write(f"{key}: [{low}, {high}]\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("参数说明:\n")
            f.write("-" * 40 + "\n")
            f.write("mcts_simulations: MCTS每次搜索的模拟次数\n")
            f.write("exploration_weight: UCB公式中的探索权重(c)\n")
            f.write("critic_lr: Critic网络的学习率\n")
            f.write("pi_lr: Policy网络的学习率\n")
            f.write("entropy_cost: 熵正则化系数\n")
            f.write("batch_size: 训练批次大小\n")
            f.write("num_critic_before_pi: 几次Critic更新后更新一次Policy\n")
            f.write("rollout_limit: rollout模拟的最大步数\n")

        print(f"📄 报告已保存到: {report_file}")


class FullTrainingOptimizer(AlphaGoBayesianOptimizer):
    """完整训练优化的扩展版本"""

    def __init__(self,
                 opponent_pool_dir="./opponent_pool",
                 num_eval_games=15,
                 num_train_games=50,
                 log_dir="./bayesian_opt_full_logs"):

        super().__init__(num_eval_games, log_dir)
        self.opponent_pool_dir = opponent_pool_dir
        self.num_train_games = num_train_games

        # 扩展参数空间
        self.pbounds.update({
            'gamma': (0.9, 0.999),  # 折扣因子
            'learning_rate_decay': (0.9, 1.0),  # 学习率衰减
            'temperature': (0.1, 1.0),  # 温度参数（探索）
            'mcts_temperature': (0.1, 1.0),  # MCTS温度
        })

    def full_training_evaluation(self, params, training_iterations=3):
        """完整训练评估（使用对手池）"""

        # 创建对手池
        opponent_pool = OpponentPool(
            pool_dir=self.opponent_pool_dir,
            max_size=5
        )

        # 创建agent
        deep_agent, deep_sess = self.create_agent_with_params(params, "deep")
        rollout_agent, rollout_sess = self.create_agent_with_params(params, "rollout")

        # 训练循环
        training_scores = []

        for iteration in range(training_iterations):
            print(f"  训练迭代 {iteration + 1}/{training_iterations}")

            # 训练几局
            iteration_results = []
            for _ in range(self.num_train_games // training_iterations):
                # 与随机对手或对手池对手对战
                result = self.play_evaluation_game(
                    deep_agent,
                    opponent="random"
                )
                iteration_results.append(result)

            # 计算迭代得分
            iteration_score = np.mean([1 if r > 0 else 0 for r in iteration_results])
            training_scores.append(iteration_score)

        # 评估
        mcts = self.create_mcts_with_params(deep_agent, rollout_agent, params)

        eval_results = []
        for i in range(self.num_eval_games):
            result = self.play_evaluation_game(
                deep_agent,
                mcts=mcts,
                opponent="random"
            )
            eval_results.append(result)

        # 计算最终得分
        final_score = np.mean(training_scores) * 50 + np.mean(eval_results) * 50

        # 清理
        deep_sess.close()
        rollout_sess.close()
        tf.reset_default_graph()

        return final_score

    def evaluate_parameters(self, **kwargs):
        """完整训练评估的目标函数"""
        params = self.convert_parameters(kwargs)

        # 添加扩展参数
        params['gamma'] = kwargs.get('gamma', 0.99)
        params['temperature'] = kwargs.get('temperature', 0.8)
        params['mcts_temperature'] = kwargs.get('mcts_temperature', 0.5)

        print(f"\n🔬 测试参数:")
        for key in ['mcts_simulations', 'exploration_weight', 'critic_lr',
                    'pi_lr', 'entropy_cost', 'batch_size']:
            print(f"  {key}: {params[key]}")

        try:
            score = self.full_training_evaluation(params)
            print(f"📈 得分: {score:.4f}")
            return score
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return -100


def run_quick_optimization():
    """运行快速优化（适合初步探索）"""
    optimizer = AlphaGoBayesianOptimizer(
        num_eval_games=15,  # 减少评估游戏数以加快速度
        log_dir="./bayesian_opt_quick"
    )

    results = optimizer.optimize(
        n_iter=20,  # 减少迭代次数
        init_points=5  # 减少初始点
    )

    return results


def run_full_optimization():
    """运行完整优化（更准确但更慢）"""
    optimizer = FullTrainingOptimizer(
        num_eval_games=10,
        num_train_games=40,
        log_dir="./bayesian_opt_full"
    )

    results = optimizer.optimize(
        n_iter=30,
        init_points=8
    )

    return results


def optimize_specific_component(component="mcts"):
    """优化特定组件"""

    if component == "mcts":
        pbounds = {
            'mcts_simulations': (10, 200),
            'exploration_weight': (0.1, 2.0),
            'rollout_limit': (10, 100),
            'mcts_temperature': (0.1, 1.0),
        }
    elif component == "rl":
        pbounds = {
            'critic_lr': (1e-5, 1e-2),
            'pi_lr': (1e-6, 1e-3),
            'entropy_cost': (0.0, 0.1),
            'batch_size_log': (4, 7),
            'num_critic_before_pi': (1, 16),
            'gamma': (0.9, 0.999),
        }
    else:
        raise ValueError("component must be 'mcts' or 'rl'")

    # 创建优化器
    optimizer = BayesianOptimization(
        f=lambda **kwargs: evaluate_component(component, kwargs),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    print(f"🎯 优化 {component.upper()} 参数")
    optimizer.maximize(init_points=5, n_iter=15)

    return optimizer.max


def evaluate_component(component, params):
    """评估特定组件"""
    # 简化评估
    base_params = {
        'mcts_simulations': 50,
        'exploration_weight': 1.0,
        'critic_lr': 0.001,
        'pi_lr': 0.0001,
        'entropy_cost': 0.01,
        'batch_size': 32,
        'num_critic_before_pi': 8,
        'rollout_limit': 50,
    }

    # 更新参数
    if component == "mcts":
        for key in ['mcts_simulations', 'exploration_weight', 'rollout_limit']:
            if key in params:
                base_params[key] = params[key]
    elif component == "rl":
        for key in ['critic_lr', 'pi_lr', 'entropy_cost', 'batch_size_log', 'num_critic_before_pi']:
            if key in params:
                if key == 'batch_size_log':
                    base_params['batch_size'] = int(2 ** params[key])
                else:
                    base_params[key] = params[key]

    # 简单评估
    optimizer = AlphaGoBayesianOptimizer(num_eval_games=10)
    score = optimizer.quick_train_and_evaluate(base_params)

    return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mini AlphaGo 贝叶斯优化调参")
    parser.add_argument("--mode", choices=["quick", "full", "mcts", "rl"],
                        default="quick", help="优化模式")
    parser.add_argument("--iterations", type=int, default=20, help="迭代次数")
    parser.add_argument("--eval-games", type=int, default=15, help="评估游戏数")

    args = parser.parse_args()

    if args.mode == "quick":
        print("🚀 快速优化模式")
        run_quick_optimization()
    elif args.mode == "full":
        print("🏃 完整优化模式")
        run_full_optimization()
    elif args.mode == "mcts":
        print("🎯 优化MCTS参数")
        optimize_specific_component("mcts")
    elif args.mode == "rl":
        print("🎯 优化RL参数")
        optimize_specific_component("rl")