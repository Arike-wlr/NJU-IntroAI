import os
os.environ['BOARD_SIZE'] = '5'
import numpy as np
import tensorflow as tf
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
import json
import time
import warnings
from datetime import datetime
import gc
import sys
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent, create_policy_agent, load_policy_agent
from algorimths.mcts import MCTS, AlphaGoMCTS
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ============================================================================
# 基础评估器
# ============================================================================

class BaseEvaluator:
    """基础评估器"""

    def __init__(self, num_eval_games=10):
        self.num_eval_games = num_eval_games

    def play_game(self, agent, mcts=None, opponent="random", use_mcts=False):
        """玩一局游戏"""
        game = Position(komi=0.5)

        while not game.is_game_over():
            current_player = 0 if game.to_play == 1 else 1

            if current_player == 0:  # 我们的AI
                if use_mcts and mcts:
                    action, _ = mcts.get_best_action(
                        game,
                        current_player,
                        num_simulations=50
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
            except Exception as e:
                # 选择合法动作
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

    def evaluate_agent(self, agent, mcts=None, use_mcts=False):
        """评估agent性能"""
        results = []
        for i in range(self.num_eval_games):
            result = self.play_game(
                agent,
                mcts=mcts,
                opponent="random",
                use_mcts=use_mcts
            )
            results.append(result)

            if (i + 1) % max(1, self.num_eval_games // 5) == 0:
                print(f"    评估进度: {i + 1}/{self.num_eval_games}")

        win_rate = np.mean([1 if r > 0 else 0 for r in results])
        avg_result = np.mean(results)

        return win_rate, avg_result, results


# ============================================================================
# 快速贝叶斯优化器
# ============================================================================

class QuickBayesianOptimizer:
    """快速贝叶斯优化器 - 用于快速探索参数空间"""

    def __init__(self,
                 num_eval_games=15,
                 num_train_games=20,
                 log_dir="./bayesian_opt_quick"):

        self.num_eval_games = num_eval_games
        self.num_train_games = num_train_games
        self.log_dir = log_dir
        self.evaluator = BaseEvaluator(num_eval_games)

        os.makedirs(log_dir, exist_ok=True)

        # 参数搜索空间
        self.pbounds = {
            # MCTS参数
            'mcts_simulations': (10, 200),
            'exploration_weight': (0.1, 2.0),
            'rollout_limit': (10, 100),

            # RL算法参数
            'critic_lr_exp': (-5, -2),  # 10^x
            'pi_lr_exp': (-6, -3),
            'entropy_cost': (0.0, 0.1),
            'batch_size_exp': (4, 6.5),  # log2
            'num_critic_before_pi': (1, 16),

            # 训练参数
            'learning_rate_decay': (0.9, 1.0),
            'temperature': (0.1, 1.0),
        }

        # 创建优化器
        self.optimizer = BayesianOptimization(
            f=self.evaluate_parameters,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )

        # 最佳结果记录
        self.best_score = -float('inf')
        self.best_params = None
        self.history = []

        # 结果文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"{log_dir}/results_{timestamp}.json"
        self.log_file = f"{log_dir}/optimization_{timestamp}.log"

        # 初始化结果文件
        with open(self.results_file, 'w') as f:
            json.dump({"optimization_history": []}, f)

    def convert_parameters(self, params):
        """转换参数到合适范围"""
        converted = {}

        # 转换MCTS参数
        converted['mcts_simulations'] = int(params['mcts_simulations'])
        converted['exploration_weight'] = float(params['exploration_weight'])
        converted['rollout_limit'] = int(params['rollout_limit'])

        # 转换学习率（对数空间）
        converted['critic_lr'] = 10 ** params['critic_lr_exp']
        converted['pi_lr'] = 10 ** params['pi_lr_exp']

        # 转换其他参数
        converted['entropy_cost'] = float(params['entropy_cost'])
        converted['batch_size'] = int(2 ** params['batch_size_exp'])
        converted['num_critic_before_pi'] = int(params['num_critic_before_pi'])
        converted['learning_rate_decay'] = float(params['learning_rate_decay'])
        converted['temperature'] = float(params['temperature'])

        return converted

    def create_agents(self, params):
        """使用给定参数创建深度和浅层agent"""
        agents = {}
        sessions = {}

        # 创建深度网络
        tf.reset_default_graph()
        deep_sess = tf.Session()
        deep_agent = GoPolicyAgent(
            session=deep_sess,
            hidden_layers=[256, 256],
            loss_str="a2c"
        )
        # 设置参数
        deep_agent.agent._critic_learning_rate = params['critic_lr']
        deep_agent.agent._pi_learning_rate = params['pi_lr']
        deep_agent.agent._entropy_cost = params['entropy_cost']
        deep_agent.agent._batch_size = params['batch_size']
        deep_agent.agent._num_critic_before_pi = params['num_critic_before_pi']

        deep_sess.run(tf.global_variables_initializer())
        agents['deep'] = deep_agent
        sessions['deep'] = deep_sess

        # 创建浅层网络
        tf.reset_default_graph()
        rollout_sess = tf.Session()
        rollout_agent = GoPolicyAgent(
            session=rollout_sess,
            hidden_layers=[64],
            loss_str="a2c"
        )
        rollout_sess.run(tf.global_variables_initializer())
        agents['rollout'] = rollout_agent
        sessions['rollout'] = rollout_sess

        return agents, sessions

    def quick_training(self, deep_agent, num_games):
        """快速训练（模拟训练过程）"""
        results = []

        for i in range(num_games):
            # 模拟训练游戏
            game = Position(komi=0.5)
            while not game.is_game_over():
                # AI的回合
                if game.to_play == 1:  # 黑棋
                    action, _ = deep_agent.select_action(game, is_evaluation=False)

                    try:
                        if action == 25:
                            game = game.pass_move(mutate=False)
                        else:
                            point = coords.from_flat(action)
                            game = game.play_move(point, mutate=False)
                    except Exception:
                        pass
                else:  # 白棋（随机对手）
                    legal_moves = game.all_legal_moves()
                    legal_actions = np.where(legal_moves == 1)[0]
                    if len(legal_actions) == 0:
                        action = 25
                    else:
                        action = np.random.choice(legal_actions)

                    try:
                        if action == 25:
                            game = game.pass_move(mutate=False)
                        else:
                            point = coords.from_flat(action)
                            game = game.play_move(point, mutate=False)
                    except Exception:
                        pass

            result = game.result()
            results.append(result)

            # 每5局打印一次进度
            if (i + 1) % max(1, num_games // 4) == 0:
                print(f"    训练进度: {i + 1}/{num_games}")

        return results

    def evaluate_parameters(self, **kwargs):
        """贝叶斯优化的目标函数"""
        try:
            # 转换参数
            params = self.convert_parameters(kwargs)
            print(f"🔬 测试参数组合 {len(self.history) + 1}:")
            # 打印关键参数
            key_params = ['mcts_simulations', 'exploration_weight', 'critic_lr',
                          'pi_lr', 'entropy_cost', 'batch_size']
            for key in key_params:
                print(f"  {key}: {params[key]}")

            # 创建agent
            print("\n🔄 创建网络...")
            agents, sessions = self.create_agents(params)

            # 快速训练
            print(f"🎮 快速训练 ({self.num_train_games}局)...")
            train_results = self.quick_training(agents['deep'], self.num_train_games)
            train_win_rate = np.mean([1 if r > 0 else 0 for r in train_results])

            # 创建MCTS
            print("🧠 创建MCTS...")
            mcts = AlphaGoMCTS(
                deep_policy_agent=None,
                rollout_policy_agent=None,
                exploration_weight=params['exploration_weight'],
                simulation_limit=params['rollout_limit']
            )
            mcts.deep_policy = agents['deep']
            mcts.rollout_policy = agents['rollout']

            # 评估（不使用MCTS）
            print(f"📊 评估网络性能 ({self.num_eval_games}局)...")
            win_rate_no_mcts, avg_result_no_mcts, _ = self.evaluator.evaluate_agent(
                agents['deep'], mcts=None, use_mcts=False
            )

            # 评估（使用MCTS）
            print(f"🤖 评估MCTS性能 ({self.num_eval_games}局)...")
            win_rate_mcts, avg_result_mcts, _ = self.evaluator.evaluate_agent(
                agents['deep'], mcts=mcts, use_mcts=True
            )

            # 计算得分
            # 权重：训练胜率(0.2) + 网络评估(0.3) + MCTS评估(0.5)
            score = (train_win_rate * 0.2 +
                     win_rate_no_mcts * 0.3 +
                     win_rate_mcts * 0.5) * 100

            # 添加额外奖励：MCTS提升效果
            mcts_improvement = max(0, win_rate_mcts - win_rate_no_mcts)
            score += mcts_improvement * 50

            # 添加额外奖励：稳定性（方差小）
            all_results = train_results + [win_rate_no_mcts, win_rate_mcts]
            stability = 1.0 - np.std(all_results)
            score += stability * 10

            # 清理资源
            print("🧹 清理资源...")
            for sess in sessions.values():
                sess.close()
            tf.reset_default_graph()
            gc.collect()

            # 记录结果
            result_entry = {
                'params': params,
                'raw_params': kwargs,
                'score': float(score),
                'train_win_rate': float(train_win_rate),
                'win_rate_no_mcts': float(win_rate_no_mcts),
                'win_rate_mcts': float(win_rate_mcts),
                'avg_result_no_mcts': float(avg_result_no_mcts),
                'avg_result_mcts': float(avg_result_mcts),
                'timestamp': datetime.now().isoformat()
            }

            self.history.append(result_entry)

            # 保存到文件
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            data['optimization_history'].append(result_entry)
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)

            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

                # 保存最佳参数
                best_params_file = f"{self.log_dir}/best_params.json"
                with open(best_params_file, 'w') as f:
                    json.dump({
                        'score': float(score),
                        'params': params,
                        'metrics': {
                            'train_win_rate': float(train_win_rate),
                            'win_rate_no_mcts': float(win_rate_no_mcts),
                            'win_rate_mcts': float(win_rate_mcts)
                        },
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                print(f"\n🎉 新的最佳得分: {score:.4f}")

            print(f"\n📈 得分详情:")
            print(f"  训练胜率: {train_win_rate:.2%}")
            print(f"  网络胜率: {win_rate_no_mcts:.2%}")
            print(f"  MCTS胜率: {win_rate_mcts:.2%}")
            print(f"  MCTS提升: {mcts_improvement:.2%}")
            print(f"  最终得分: {score:.4f}")
            print('=' * 60)

            return score

        except Exception as e:
            print(f"评估失败: {e}")
            import traceback
            traceback.print_exc()
            return -10.0

    def optimize(self, n_iter=25, init_points=8):
        """运行贝叶斯优化"""

        print("=" * 70)
        print("🎯 Mini AlphaGo 贝叶斯优化调参")
        print("=" * 70)
        print(f"优化模式: 快速探索")
        print(f"迭代次数: {n_iter}")
        print(f"初始点: {init_points}")
        print(f"训练游戏数: {self.num_train_games}")
        print(f"评估游戏数: {self.num_eval_games}")
        print(f"日志目录: {self.log_dir}")
        print("=" * 70)

        # 创建日志记录器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = JSONLogger(path=f"{self.log_dir}/optimization_{timestamp}.json")
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # 运行优化
        start_time = time.time()

        try:
            self.optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )
        except KeyboardInterrupt:
            print("优化被用户中断")
        except Exception as e:
            print(f"优化过程中出错: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # 打印最佳结果
        self.print_final_results(elapsed_time)

        return self.optimizer.max

    def print_final_results(self, elapsed_time):
        """打印最终结果"""
        if hasattr(self.optimizer, 'max'):
            best_result = self.optimizer.max
            print(f"最佳得分: {best_result['target']:.4f}")
            print("最佳参数配置:")

            # 转换并打印参数
            best_params = self.convert_parameters(best_result['params'])

            # 按类别分组打印
            print("MCTS参数:")
            mcts_params = ['mcts_simulations', 'exploration_weight', 'rollout_limit']
            for key in mcts_params:
                print(f"  {key:25s}: {best_params[key]}")

            print("RL算法参数:")
            rl_params = ['critic_lr', 'pi_lr', 'entropy_cost', 'batch_size',
                         'num_critic_before_pi']
            for key in rl_params:
                print(f"  {key:25s}: {best_params[key]}")

            print("训练参数:")
            train_params = ['learning_rate_decay', 'temperature']
            for key in train_params:
                print(f"  {key:25s}: {best_params[key]}")

        print(f"总耗时: {elapsed_time:.1f}秒")
        print(f"测试参数组合数: {len(self.history)}")
        print(f"结果已保存到: {self.results_file}")

        # 生成最终报告
        self.generate_final_report(elapsed_time)

    def generate_final_report(self, elapsed_time):
        """生成最终报告"""
        report_file = f"{self.log_dir}/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Mini AlphaGo 贝叶斯优化调参报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"优化模式: 快速探索\n")
            f.write(f"总耗时: {elapsed_time:.1f}秒\n")
            f.write(f"测试参数组合数: {len(self.history)}\n\n")

            f.write("🏆 最佳参数配置:\n")
            f.write("-" * 50 + "\n")

            if self.best_params:
                # 按类别分组
                categories = {
                    "MCTS参数": ['mcts_simulations', 'exploration_weight', 'rollout_limit'],
                    "RL算法参数": ['critic_lr', 'pi_lr', 'entropy_cost', 'batch_size',
                                   'num_critic_before_pi'],
                    "训练参数": ['learning_rate_decay', 'temperature']
                }

                for category, params_list in categories.items():
                    f.write(f"\n{category}:\n")
                    for param in params_list:
                        if param in self.best_params:
                            f.write(f"  {param:25s}: {self.best_params[param]}\n")

                f.write(f"最佳得分: {self.best_score:.4f}\n")
            f.write("参数说明:\n")
            param_descriptions = {
                'mcts_simulations': 'MCTS每次搜索的模拟次数',
                'exploration_weight': 'UCB公式中的探索权重(c值)',
                'rollout_limit': '快速走子模拟的最大步数',
                'critic_lr': 'Critic网络的学习率',
                'pi_lr': 'Policy网络的学习率',
                'entropy_cost': '策略熵的正则化系数',
                'batch_size': '训练批次大小',
                'num_critic_before_pi': '几次Critic更新后更新一次Policy',
                'learning_rate_decay': '学习率衰减因子',
                'temperature': '动作选择的温度参数'
            }

            for param, desc in param_descriptions.items():
                f.write(f"{param:25s}: {desc}\n")

        print(f"完整报告已保存到: {report_file}")


# ============================================================================
# 完整贝叶斯优化器
# ============================================================================

class FullBayesianOptimizer(QuickBayesianOptimizer):
    """完整贝叶斯优化器 - 更准确但更慢"""

    def __init__(self,
                 num_eval_games=20,
                 num_train_games=50,
                 log_dir="./bayesian_opt_full"):

        super().__init__(num_eval_games, num_train_games, log_dir)

        # 扩展参数空间
        self.pbounds.update({
            'gamma': (0.95, 0.999),  # 折扣因子
            'mcts_temperature': (0.1, 1.0),  # MCTS温度参数
            'value_loss_weight': (0.1, 1.0),  # 价值损失权重
            'clip_param': (0.1, 0.3),  # PPO裁剪参数（如使用）
        })

    def evaluate_parameters(self, **kwargs):
        """更完整的评估函数"""
        try:
            # 转换参数
            params = self.convert_parameters(kwargs)

            # 添加扩展参数
            params['gamma'] = kwargs.get('gamma', 0.99)
            params['mcts_temperature'] = kwargs.get('mcts_temperature', 0.8)
            params['value_loss_weight'] = kwargs.get('value_loss_weight', 0.5)
            params['clip_param'] = kwargs.get('clip_param', 0.2)

            print(f"\n{'=' * 70}")
            print(f"🔬 完整测试参数组合 {len(self.history) + 1}:")
            print("-" * 70)

            # 打印所有参数
            for key in sorted(params.keys()):
                print(f"  {key:25s}: {params[key]}")

            # 创建agent
            print("\n🔄 创建网络...")
            agents, sessions = self.create_agents(params)

            # 更长的训练
            print(f"🎮 完整训练 ({self.num_train_games * 2}局)...")
            train_results = self.quick_training(agents['deep'], self.num_train_games * 2)
            train_win_rate = np.mean([1 if r > 0 else 0 for r in train_results])

            # 创建MCTS（使用温度参数）
            print("创建MCTS...")
            mcts = AlphaGoMCTS(
                deep_policy_agent=None,
                rollout_policy_agent=None,
                exploration_weight=params['exploration_weight'],
                simulation_limit=params['rollout_limit']
            )
            mcts.deep_policy = agents['deep']
            mcts.rollout_policy = agents['rollout']

            # 更多评估游戏
            eval_games = self.num_eval_games * 2

            print(f"评估网络性能 ({eval_games}局)...")
            win_rate_no_mcts, avg_result_no_mcts, _ = self.evaluator.evaluate_agent(
                agents['deep'], mcts=None, use_mcts=False
            )

            print(f"评估MCTS性能 ({eval_games}局)...")
            win_rate_mcts, avg_result_mcts, _ = self.evaluator.evaluate_agent(
                agents['deep'], mcts=mcts, use_mcts=True
            )

            # 计算更复杂的得分
            # 权重调整：更注重MCTS性能
            score = (train_win_rate * 0.15 +
                     win_rate_no_mcts * 0.25 +
                     win_rate_mcts * 0.6) * 100

            # MCTS提升效果
            mcts_improvement = max(0, win_rate_mcts - win_rate_no_mcts)
            score += mcts_improvement * 60

            # 训练稳定性
            stability = 1.0 - np.std(train_results)
            score += stability * 15

            # 参数合理性奖励
            param_bonus = self.calculate_param_bonus(params)
            score += param_bonus

            # 清理资源
            print("🧹 清理资源...")
            for sess in sessions.values():
                sess.close()
            tf.reset_default_graph()
            gc.collect()

            # 记录结果
            result_entry = {
                'params': params,
                'raw_params': kwargs,
                'score': float(score),
                'train_win_rate': float(train_win_rate),
                'win_rate_no_mcts': float(win_rate_no_mcts),
                'win_rate_mcts': float(win_rate_mcts),
                'avg_result_no_mcts': float(avg_result_no_mcts),
                'avg_result_mcts': float(avg_result_mcts),
                'mcts_improvement': float(mcts_improvement),
                'timestamp': datetime.now().isoformat()
            }

            self.history.append(result_entry)

            # 保存到文件
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            data['optimization_history'].append(result_entry)
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)

            # 更新最佳结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

                best_params_file = f"{self.log_dir}/best_params.json"
                with open(best_params_file, 'w') as f:
                    json.dump({
                        'score': float(score),
                        'params': params,
                        'metrics': {
                            'train_win_rate': float(train_win_rate),
                            'win_rate_no_mcts': float(win_rate_no_mcts),
                            'win_rate_mcts': float(win_rate_mcts),
                            'mcts_improvement': float(mcts_improvement)
                        },
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                print(f"新的最佳得分: {score:.4f}")

            print(f"得分详情:")
            print(f"  训练胜率: {train_win_rate:.2%}")
            print(f"  网络胜率: {win_rate_no_mcts:.2%}")
            print(f"  MCTS胜率: {win_rate_mcts:.2%}")
            print(f"  MCTS提升: {mcts_improvement:.2%}")
            print(f"  参数奖励: {param_bonus:.2f}")
            print(f"  最终得分: {score:.4f}")
            print('=' * 70)

            return score

        except Exception as e:
            print(f"评估失败: {e}")
            import traceback
            traceback.print_exc()
            return -10.0

    def calculate_param_bonus(self, params):
        """计算参数合理性奖励"""
        bonus = 0

        # 学习率比例合理
        if 0.05 <= params['pi_lr'] / params['critic_lr'] <= 0.2:
            bonus += 5

        # 批次大小合理
        if 16 <= params['batch_size'] <= 128:
            bonus += 3

        # MCTS模拟次数合理
        if 30 <= params['mcts_simulations'] <= 150:
            bonus += 3

        # 探索权重合理
        if 0.5 <= params['exploration_weight'] <= 1.5:
            bonus += 2

        return bonus


# ============================================================================
# 组件特定优化器
# ============================================================================

class MCTSOptimizer:
    """专门优化MCTS参数"""

    def __init__(self, num_eval_games=20, log_dir="./bayesian_opt_mcts"):
        self.num_eval_games = num_eval_games
        self.log_dir = log_dir
        self.evaluator = BaseEvaluator(num_eval_games)

        os.makedirs(log_dir, exist_ok=True)

        # MCTS特定参数空间
        self.pbounds = {
            'mcts_simulations': (10, 300),
            'exploration_weight': (0.1, 2.5),
            'rollout_limit': (10, 150),
            'mcts_temperature': (0.01, 1.0),
            'ucb_constant': (0.5, 2.5),  # UCB常数
        }

        self.optimizer = BayesianOptimization(
            f=self.evaluate_mcts_params,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )

        self.best_score = -float('inf')
        self.best_params = None

    def evaluate_mcts_params(self, **kwargs):
        """评估MCTS参数"""
        try:
            params = {
                'mcts_simulations': int(kwargs['mcts_simulations']),
                'exploration_weight': float(kwargs['exploration_weight']),
                'rollout_limit': int(kwargs['rollout_limit']),
                'mcts_temperature': float(kwargs['mcts_temperature']),
                'ucb_constant': float(kwargs['ucb_constant']),
            }

            print(f"\n🔬 测试MCTS参数:")
            for key, value in params.items():
                print(f"  {key}: {value}")

            # 创建基础agent（使用默认参数）
            print("\n🔄 创建基础网络...")
            tf.reset_default_graph()
            deep_sess = tf.Session()
            deep_agent = GoPolicyAgent(
                session=deep_sess,
                hidden_layers=[256, 256],
                loss_str="a2c"
            )
            deep_sess.run(tf.global_variables_initializer())

            tf.reset_default_graph()
            rollout_sess = tf.Session()
            rollout_agent = GoPolicyAgent(
                session=rollout_sess,
                hidden_layers=[64],
                loss_str="a2c"
            )
            rollout_sess.run(tf.global_variables_initializer())

            # 创建MCTS
            print("创建MCTS...")
            mcts = AlphaGoMCTS(
                deep_policy_agent=None,
                rollout_policy_agent=None,
                exploration_weight=params['exploration_weight'],
                simulation_limit=params['rollout_limit']
            )
            mcts.deep_policy = deep_agent
            mcts.rollout_policy = rollout_agent

            # 评估网络基础性能
            print(f"评估基础网络性能 ({self.num_eval_games}局)...")
            win_rate_no_mcts, _, _ = self.evaluator.evaluate_agent(
                deep_agent, mcts=None, use_mcts=False
            )

            # 评估MCTS性能
            print(f"评估MCTS性能 ({self.num_eval_games}局)...")
            win_rate_mcts, _, _ = self.evaluator.evaluate_agent(
                deep_agent, mcts=mcts, use_mcts=True
            )

            # 计算MCTS提升效果
            improvement = win_rate_mcts - win_rate_no_mcts
            score = win_rate_mcts * 100 + improvement * 50

            # 清理资源
            deep_sess.close()
            rollout_sess.close()
            tf.reset_default_graph()
            gc.collect()

            print(f"MCTS性能:")
            print(f"  基础胜率: {win_rate_no_mcts:.2%}")
            print(f"  MCTS胜率: {win_rate_mcts:.2%}")
            print(f"  提升效果: {improvement:.2%}")
            print(f"  最终得分: {score:.4f}")

            # 保存结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

                with open(f"{self.log_dir}/best_mcts_params.json", 'w') as f:
                    json.dump({
                        'score': float(score),
                        'params': params,
                        'win_rate_no_mcts': float(win_rate_no_mcts),
                        'win_rate_mcts': float(win_rate_mcts),
                        'improvement': float(improvement),
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

            return score

        except Exception as e:
            print(f"\n❌ MCTS评估失败: {e}")
            return -10.0

    def optimize(self, n_iter=20, init_points=6):
        """优化MCTS参数"""
        print("=" * 60)
        print("🎯 专门优化MCTS参数")
        print("=" * 60)

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        print(f"\n🏆 MCTS最佳得分: {self.best_score:.4f}")
        print("🎯 MCTS最佳参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        return self.optimizer.max


class RLOptimizer:
    """专门优化RL参数"""

    def __init__(self, num_eval_games=20, log_dir="./bayesian_opt_rl"):
        self.num_eval_games = num_eval_games
        self.log_dir = log_dir
        self.evaluator = BaseEvaluator(num_eval_games)

        os.makedirs(log_dir, exist_ok=True)

        # RL特定参数空间
        self.pbounds = {
            'critic_lr_exp': (-5, -2),
            'pi_lr_exp': (-6, -3),
            'entropy_cost': (0.0, 0.1),
            'batch_size_exp': (4, 7),
            'num_critic_before_pi': (1, 20),
            'gamma': (0.9, 0.999),
            'value_loss_weight': (0.1, 1.0),
            'learning_rate_decay': (0.9, 1.0),
        }

        self.optimizer = BayesianOptimization(
            f=self.evaluate_rl_params,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )

        self.best_score = -float('inf')
        self.best_params = None

    def evaluate_rl_params(self, **kwargs):
        """评估RL参数"""
        try:
            params = {
                'critic_lr': 10 ** kwargs['critic_lr_exp'],
                'pi_lr': 10 ** kwargs['pi_lr_exp'],
                'entropy_cost': kwargs['entropy_cost'],
                'batch_size': int(2 ** kwargs['batch_size_exp']),
                'num_critic_before_pi': int(kwargs['num_critic_before_pi']),
                'gamma': kwargs['gamma'],
                'value_loss_weight': kwargs['value_loss_weight'],
                'learning_rate_decay': kwargs['learning_rate_decay'],
            }

            print(f"\n🔬 测试RL参数:")
            for key, value in params.items():
                print(f"  {key}: {value}")

            # 创建agent
            print("\n🔄 创建网络...")
            tf.reset_default_graph()
            sess = tf.Session()
            agent = GoPolicyAgent(
                session=sess,
                hidden_layers=[256, 256],
                loss_str="a2c"
            )

            # 设置RL参数
            agent.agent._critic_learning_rate = params['critic_lr']
            agent.agent._pi_learning_rate = params['pi_lr']
            agent.agent._entropy_cost = params['entropy_cost']
            agent.agent._batch_size = params['batch_size']
            agent.agent._num_critic_before_pi = params['num_critic_before_pi']

            sess.run(tf.global_variables_initializer())

            # 快速训练评估
            print(f"🎮 快速训练评估 ({self.num_eval_games * 2}局)...")

            results = []
            for i in range(self.num_eval_games * 2):
                # 模拟训练游戏
                game = Position(komi=0.5)
                while not game.is_game_over():
                    if game.to_play == 1:  # 黑棋
                        action, _ = agent.select_action(game, is_evaluation=False)
                        try:
                            if action == 25:
                                game = game.pass_move(mutate=False)
                            else:
                                point = coords.from_flat(action)
                                game = game.play_move(point, mutate=False)
                        except Exception:
                            pass
                    else:  # 白棋
                        legal_moves = game.all_legal_moves()
                        legal_actions = np.where(legal_moves == 1)[0]
                        if len(legal_actions) == 0:
                            action = 25
                        else:
                            action = np.random.choice(legal_actions)

                        try:
                            if action == 25:
                                game = game.pass_move(mutate=False)
                            else:
                                point = coords.from_flat(action)
                                game = game.play_move(point, mutate=False)
                        except Exception:
                            pass

                result = game.result()
                results.append(result)

                if (i + 1) % max(1, (self.num_eval_games * 2) // 5) == 0:
                    print(f"    进度: {i + 1}/{self.num_eval_games * 2}")

            win_rate = np.mean([1 if r > 0 else 0 for r in results])
            avg_result = np.mean(results)

            # 计算得分
            score = win_rate * 100 + avg_result * 10

            # 参数合理性奖励
            bonus = 0
            if 0.05 <= params['pi_lr'] / params['critic_lr'] <= 0.2:
                bonus += 5
            if 16 <= params['batch_size'] <= 128:
                bonus += 3

            score += bonus

            # 清理资源
            sess.close()
            tf.reset_default_graph()
            gc.collect()

            print(f"\n📈 RL网络性能:")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  平均结果: {avg_result:.3f}")
            print(f"  参数奖励: {bonus:.1f}")
            print(f"  最终得分: {score:.4f}")

            # 保存结果
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

                with open(f"{self.log_dir}/best_rl_params.json", 'w') as f:
                    json.dump({
                        'score': float(score),
                        'params': params,
                        'win_rate': float(win_rate),
                        'avg_result': float(avg_result),
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

            return score

        except Exception as e:
            print(f"\n❌ RL评估失败: {e}")
            return -10.0

    def optimize(self, n_iter=20, init_points=6):
        """优化RL参数"""
        print("=" * 60)
        print("🎯 专门优化RL参数")
        print("=" * 60)

        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        print(f"\n🏆 RL最佳得分: {self.best_score:.4f}")
        print("🎯 RL最佳参数:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        return self.optimizer.max


# ============================================================================
# 分析工具
# ============================================================================

class OptimizationAnalyzer:
    """优化结果分析工具"""

    @staticmethod
    def load_results(log_dir):
        """加载优化结果"""
        results_file = None
        for file in os.listdir(log_dir):
            if file.startswith("results_") and file.endswith(".json"):
                results_file = f"{log_dir}/{file}"
                break

        if not results_file:
            print(f"❌ 在 {log_dir} 中找不到结果文件")
            return None

        with open(results_file, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def analyze_correlations(data, top_n=10):
        """分析参数与得分的相关性"""
        if not data or 'optimization_history' not in data:
            print("❌ 数据格式不正确")
            return

        history = data['optimization_history']
        if not history:
            print("❌ 没有历史数据")
            return

        # 收集所有参数
        all_params = set()
        for entry in history:
            all_params.update(entry['params'].keys())

        # 分析每个参数的影响
        print("\n📊 参数影响分析:")
        print("-" * 60)

        for param in sorted(all_params):
            if param == 'timestamp':
                continue

            values = []
            scores = []

            for entry in history:
                if param in entry['params']:
                    values.append(entry['params'][param])
                    scores.append(entry['score'])

            if len(values) > 3:
                # 计算相关性
                correlation = np.corrcoef(values, scores)[0, 1]

                # 找到最佳值范围
                sorted_indices = np.argsort(scores)[-5:]  # 前5个最佳得分
                best_values = [values[i] for i in sorted_indices]

                if isinstance(best_values[0], (int, float)):
                    avg_best = np.mean(best_values)
                    std_best = np.std(best_values)
                    print(f"{param:25s}: 相关性={correlation:6.3f} | 最佳值={avg_best:8.3f}±{std_best:6.3f}")
                else:
                    print(f"{param:25s}: 相关性={correlation:6.3f}")

    @staticmethod
    def print_top_results(data, top_n=5):
        """打印最佳结果"""
        if not data or 'optimization_history' not in data:
            return

        history = data['optimization_history']
        sorted_history = sorted(history, key=lambda x: x['score'], reverse=True)

        print(f"\n🏆 前{top_n}个最佳配置:")
        print("=" * 70)

        for i, entry in enumerate(sorted_history[:top_n]):
            print(f"\n第{i + 1}名 | 得分: {entry['score']:.4f}")
            print("-" * 40)

            # 按类别显示参数
            categories = {
                "MCTS参数": ['mcts_simulations', 'exploration_weight', 'rollout_limit'],
                "RL参数": ['critic_lr', 'pi_lr', 'entropy_cost', 'batch_size',
                           'num_critic_before_pi'],
                "性能指标": ['train_win_rate', 'win_rate_no_mcts', 'win_rate_mcts']
            }

            for category, params in categories.items():
                print(f"\n{category}:")
                for param in params:
                    if param in entry['params']:
                        print(f"  {param:25s}: {entry['params'][param]}")
                    elif param in entry:
                        print(f"  {param:25s}: {entry[param]}")

            print(f"时间: {entry.get('timestamp', 'N/A')}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mini AlphaGo 贝叶斯优化调参系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python bayesian_optimization.py --mode quick --iterations 20
  python bayesian_optimization.py --mode full --eval-games 15
  python bayesian_optimization.py --mode mcts --iterations 15
  python bayesian_optimization.py --mode rl --iterations 15
  python bayesian_optimization.py --analyze --log-dir ./bayesian_opt_quick

参数说明:
  quick: 快速优化模式，适合初步探索
  full: 完整优化模式，更准确但更慢
  mcts: 专门优化MCTS参数
  rl: 专门优化RL算法参数
  analyze: 分析已有的优化结果
        """
    )

    parser.add_argument("--mode",
                        choices=["quick", "full", "mcts", "rl", "analyze"],
                        default="quick",
                        help="优化模式")

    parser.add_argument("--iterations", type=int, default=20,
                        help="贝叶斯优化迭代次数")

    parser.add_argument("--init-points", type=int, default=8,
                        help="初始随机采样点数量")

    parser.add_argument("--eval-games", type=int, default=15,
                        help="每个参数配置的评估游戏数量")

    parser.add_argument("--train-games", type=int, default=20,
                        help="每个参数配置的训练游戏数量")

    parser.add_argument("--log-dir", type=str,
                        help="日志目录路径")

    parser.add_argument("--top-n", type=int, default=5,
                        help="分析时显示的最佳配置数量")

    args = parser.parse_args()

    # 设置默认日志目录
    if not args.log_dir:
        if args.mode == "quick":
            args.log_dir = "./bayesian_opt_quick"
        elif args.mode == "full":
            args.log_dir = "./bayesian_opt_full"
        elif args.mode == "mcts":
            args.log_dir = "./bayesian_opt_mcts"
        elif args.mode == "rl":
            args.log_dir = "./bayesian_opt_rl"
        elif args.mode == "analyze":
            args.log_dir = "./bayesian_opt_quick"

    # 运行优化
    if args.mode == "analyze":
        print("🔍 分析优化结果...")
        analyzer = OptimizationAnalyzer()
        data = analyzer.load_results(args.log_dir)
        if data:
            analyzer.analyze_correlations(data)
            analyzer.print_top_results(data, args.top_n)
        return

    print("=" * 70)
    print("🤖 Mini AlphaGo 贝叶斯优化调参系统")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"迭代次数: {args.iterations}")
    print(f"初始点: {args.init_points}")
    print(f"评估游戏数: {args.eval_games}")
    print(f"训练游戏数: {args.train_games}")
    print(f"日志目录: {args.log_dir}")
    print("=" * 70)

    # 根据模式选择优化器
    if args.mode == "quick":
        optimizer = QuickBayesianOptimizer(
            num_eval_games=args.eval_games,
            num_train_games=args.train_games,
            log_dir=args.log_dir
        )
        optimizer.optimize(n_iter=args.iterations, init_points=args.init_points)

    elif args.mode == "full":
        optimizer = FullBayesianOptimizer(
            num_eval_games=args.eval_games,
            num_train_games=args.train_games,
            log_dir=args.log_dir
        )
        optimizer.optimize(n_iter=args.iterations, init_points=args.init_points)

    elif args.mode == "mcts":
        optimizer = MCTSOptimizer(
            num_eval_games=args.eval_games,
            log_dir=args.log_dir
        )
        optimizer.optimize(n_iter=args.iterations, init_points=args.init_points)

    elif args.mode == "rl":
        optimizer = RLOptimizer(
            num_eval_games=args.eval_games,
            log_dir=args.log_dir
        )
        optimizer.optimize(n_iter=args.iterations, init_points=args.init_points)

    print("优化完成! 最佳参数配置已保存到日志目录。")


if __name__ == "__main__":
    main()