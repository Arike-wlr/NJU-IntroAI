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
import copy
from environment.go import Position
from environment import coords
from agent.agent import GoPolicyAgent
from algorimths.mcts import MCTS, MCTSNode
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CustomAlphaGoMCTS(MCTS):
    """自定义AlphaGo MCTS，直接使用agent对象而不是文件路径"""
    def __init__(self, deep_policy_agent, rollout_policy_agent,
                 exploration_weight=1.0, simulation_limit=50):
        super().__init__(exploration_weight, simulation_limit)
        self.deep_policy = deep_policy_agent
        self.rollout_policy = rollout_policy_agent
        self.board_size = 5

    def _expand(self, node):
        """扩展：使用深度策略网络获取先验概率"""
        legal_moves_mask = node.position.all_legal_moves()
        legal_actions = np.where(legal_moves_mask == 1)[0]

        if len(legal_actions) == 0:
            return

        # 使用深度网络预测先验概率
        prior_probs = self.deep_policy.get_action_probs(node.position)

        for action in legal_actions:
            try:
                new_position = copy.deepcopy(node.position)

                if action == self.board_size * self.board_size:  # pass
                    new_position = new_position.pass_move(mutate=True)
                else:
                    from environment import coords
                    move = coords.from_flat(action)
                    new_position = new_position.play_move(move, mutate=True)

                next_player = 0 if new_position.to_play == 1 else 1

                prior = prior_probs[action]
                child = MCTSNode(new_position, next_player, parent=node, prior=prior)
                node.children[action] = child

            except Exception as e:
                print(f"扩展时出错: {e}")
                continue

    def _simulate(self, node):
        """模拟：使用快速走子网络"""
        current_position = copy.deepcopy(node.position)
        steps = 0

        while not current_position.is_game_over() and steps < self.simulation_limit:
            # 使用浅层网络预测
            action_probs = self.rollout_policy.get_action_probs(current_position)

            legal_moves = current_position.all_legal_moves()
            legal_actions = np.where(legal_moves == 1)[0]

            if len(legal_actions) == 0:
                break

            # 根据概率选择动作
            legal_probs = action_probs[legal_actions]
            legal_probs = legal_probs / (legal_probs.sum() + 1e-8)

            temperature = 1.0
            if temperature == 0:
                action_idx = np.argmax(legal_probs)
            else:
                log_probs = np.log(legal_probs + 1e-8) / temperature
                exp_log_probs = np.exp(log_probs)
                probs = exp_log_probs / exp_log_probs.sum()
                action_idx = np.random.choice(len(legal_actions), p=probs)

            action = legal_actions[action_idx]

            # 执行动作
            try:
                if action == self.board_size * self.board_size:
                    current_position = current_position.pass_move(mutate=True)
                else:
                    from environment import coords
                    move = coords.from_flat(action)
                    current_position = current_position.play_move(move, mutate=True)

                steps += 1

            except Exception as e:
                legal_moves = current_position.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]

                if len(legal_actions) == 0:
                    break

                action = np.random.choice(legal_actions)

                if action == self.board_size * self.board_size:
                    current_position = current_position.pass_move(mutate=True)
                else:
                    move = coords.from_flat(action)
                    current_position = current_position.play_move(move, mutate=True)

                steps += 1

        # 评估最终局面
        return self._evaluate_position(current_position, node.player)

    def _evaluate_position(self, position, player):
        """评估局面价值"""
        if position.is_game_over():
            result = position.result()
            if player == 0:
                value = result
            else:
                value = -result
        else:
            score = position.score()
            max_score = self.board_size * self.board_size
            value = np.clip(score / max_score, -1, 1)

            if player == 1:
                value = -value

        return value


class GoEvaluator:
    """围棋评估器"""
    def __init__(self, num_eval_games=10):
        self.num_eval_games = num_eval_games

    def play_game(self, agent, mcts=None, use_mcts=False):
        """玩一局游戏"""
        game = Position(komi=0.5)

        while not game.is_game_over():
            current_player = 0 if game.to_play == 1 else 1

            if current_player == 0:  # AI的回合
                if use_mcts and mcts:
                    action, _ = mcts.get_best_action(
                        game,
                        current_player,
                        num_simulations=50
                    )
                else:
                    action, _ = agent.select_action(game, is_evaluation=True)
            else:  # 随机对手的回合
                legal_moves = game.all_legal_moves()
                legal_actions = np.where(legal_moves == 1)[0]
                if len(legal_actions) == 0:
                    action = 25
                else:
                    action = np.random.choice(legal_actions)
            # 执行动作
            try:
                if action == 25:
                    game = game.pass_move(mutate=False)
                else:
                    point = coords.from_flat(action)
                    game = game.play_move(point, mutate=False)
            except Exception:
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

    def evaluate(self, agent, mcts=None, use_mcts=False):
        """评估agent性能"""
        results = []
        for i in range(self.num_eval_games):
            result = self.play_game(agent, mcts=mcts, use_mcts=use_mcts)
            results.append(result)

            if (i + 1) % max(1, self.num_eval_games // 5) == 0:
                print(f"评估进度: {i + 1}/{self.num_eval_games}")

        win_rate = np.mean([1 if r > 0 else 0 for r in results])
        avg_result = np.mean(results)

        return win_rate, avg_result, results


class MiniAlphaGoOptimizer:
    """Mini AlphaGo贝叶斯优化器 - 专门优化MCTS rollout次数和RL超参数"""

    def __init__(self, num_eval_games=15, num_train_games=20, log_dir="./bayesian_opt"):
        self.num_eval_games = num_eval_games
        self.num_train_games = num_train_games
        self.log_dir = log_dir
        self.evaluator = GoEvaluator(num_eval_games)

        os.makedirs(log_dir, exist_ok=True)

        # 参数搜索空间 - 专注于核心参数
        self.pbounds = {
            # MCTS参数
            'mcts_simulations': (50, 300),  # MCTS模拟次数
            'exploration_weight': (0.5, 2.0),  # UCB探索权重
            'rollout_limit': (20, 100),  # rollout最大步数
            # RL算法参数
            'critic_lr_exp': (-4, -2),  # Critic学习率 (10^-4 到 10^-2)
            'pi_lr_exp': (-5, -3),  # Policy学习率 (10^-5 到 10^-3)
            'entropy_cost': (0.001, 0.05),  # 熵正则化系数
            'batch_size_exp': (4, 6),  # 批次大小 (16 到 64)
            'num_critic_before_pi': (3, 10),  # Critic更新次数
        }

        # 创建贝叶斯优化器
        self.optimizer = BayesianOptimization(
            f=self.evaluate_parameters,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2
        )
        self.best_score = -float('inf')
        self.best_params = None
        self.history = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"{log_dir}/results_{timestamp}.json"

    def convert_parameters(self, raw_params):
        """转换参数到合适范围"""
        params = {}
        # 转换MCTS参数
        params['mcts_simulations'] = int(raw_params['mcts_simulations'])
        params['exploration_weight'] = float(raw_params['exploration_weight'])
        params['rollout_limit'] = int(raw_params['rollout_limit'])
        params['critic_lr'] = 10 ** raw_params['critic_lr_exp']
        params['pi_lr'] = 10 ** raw_params['pi_lr_exp']
        params['entropy_cost'] = float(raw_params['entropy_cost'])
        params['batch_size'] = int(2 ** raw_params['batch_size_exp'])
        params['num_critic_before_pi'] = int(raw_params['num_critic_before_pi'])
        return params

    def create_agents_with_params(self, params):
        """使用给定参数创建深度和浅层agent"""
        # 创建深度网络
        tf.reset_default_graph()
        deep_sess = tf.Session()
        deep_agent = GoPolicyAgent(
            session=deep_sess,
            hidden_layers=[256, 256],
            loss_str="a2c"
        )
        deep_agent.agent._critic_learning_rate = params['critic_lr']
        deep_agent.agent._pi_learning_rate = params['pi_lr']
        deep_agent.agent._entropy_cost = params['entropy_cost']
        deep_agent.agent._batch_size = params['batch_size']
        deep_agent.agent._num_critic_before_pi = params['num_critic_before_pi']

        deep_sess.run(tf.global_variables_initializer())

        # 创建浅层网络（固定配置）
        tf.reset_default_graph()
        rollout_sess = tf.Session()
        rollout_agent = GoPolicyAgent(
            session=rollout_sess,
            hidden_layers=[64],
            loss_str="a2c"
        )
        rollout_sess.run(tf.global_variables_initializer())

        return {
            'deep': {'agent': deep_agent, 'sess': deep_sess},
            'rollout': {'agent': rollout_agent, 'sess': rollout_sess}
        }

    def quick_training(self, agent, num_games):
        """快速训练（模拟训练过程）"""
        results = []
        for i in range(num_games):
            # 创建新游戏
            game = Position(komi=0.5)

            while not game.is_game_over():
                if game.to_play == 1:  # 黑棋（AI）
                    action, _ = agent.select_action(game, is_evaluation=False)

                    try:
                        if action == 25:
                            game = game.pass_move(mutate=False)
                        else:
                            point = coords.from_flat(action)
                            game = game.play_move(point, mutate=False)
                    except Exception:
                        # 如果动作不合法，选择第一个合法动作
                        legal_moves = game.all_legal_moves()
                        legal_actions = np.where(legal_moves == 1)[0]
                        if len(legal_actions) > 0:
                            action = legal_actions[0]
                            if action == 25:
                                game = game.pass_move(mutate=False)
                            else:
                                point = coords.from_flat(action)
                                game = game.play_move(point, mutate=False)
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

        return results

    def evaluate_parameters(self, **kwargs):
        """贝叶斯优化的目标函数"""
        try:
            # 转换参数
            params = self.convert_parameters(kwargs)
            print(f"测试参数组合 {len(self.history) + 1}:")

            # 显示关键参数
            print("MCTS参数:")
            print(f"  模拟次数: {params['mcts_simulations']}")
            print(f"  探索权重: {params['exploration_weight']:.3f}")
            print(f"  Rollout限制: {params['rollout_limit']}")

            print("\nRL算法参数:")
            print(f"  Critic学习率: {params['critic_lr']:.6f}")
            print(f"  Policy学习率: {params['pi_lr']:.6f}")
            print(f"  熵正则化: {params['entropy_cost']:.4f}")
            print(f"  批次大小: {params['batch_size']}")
            print(f"  Critic更新次数: {params['num_critic_before_pi']}")

            # 参数合理性检查
            if not self.validate_parameters(params):
                print("参数不合理，返回低分")
                return -5.0

            # 创建agent
            print("\n创建网络...")
            agents = self.create_agents_with_params(params)
            deep_agent = agents['deep']['agent']
            rollout_agent = agents['rollout']['agent']

            # 快速训练
            print(f"快速训练 ({self.num_train_games}局)...")
            train_results = self.quick_training(deep_agent, self.num_train_games)
            train_win_rate = np.mean([1 if r > 0 else 0 for r in train_results])

            # 创建MCTS
            print("创建MCTS...")
            mcts = CustomAlphaGoMCTS(
                deep_policy_agent=deep_agent,
                rollout_policy_agent=rollout_agent,
                exploration_weight=params['exploration_weight'],
                simulation_limit=params['rollout_limit']
            )
            # 评估网络基础性能（不使用MCTS）
            print(f"评估网络性能 ({self.num_eval_games}局)...")
            win_rate_no_mcts, avg_result_no_mcts, _ = self.evaluator.evaluate(
                deep_agent, mcts=None, use_mcts=False
            )

            # 评估MCTS性能
            print(f"评估MCTS性能 ({self.num_eval_games}局)...")
            win_rate_mcts, avg_result_mcts, _ = self.evaluator.evaluate(
                deep_agent, mcts=mcts, use_mcts=True
            )

            # 权重：训练胜率(0.2) + 网络评估(0.3) + MCTS评估(0.5)
            score = (train_win_rate * 0.2 +
                     win_rate_no_mcts * 0.3 +
                     win_rate_mcts * 0.5) * 100

            # 添加额外奖励：MCTS提升效果
            mcts_improvement = max(0, win_rate_mcts - win_rate_no_mcts)
            score += mcts_improvement * 50

            # 参数合理性奖励
            param_bonus = self.calculate_param_bonus(params)
            score += param_bonus

            # 清理资源
            print("清理资源...")
            for agent_type in ['deep', 'rollout']:
                agents[agent_type]['sess'].close()
            tf.reset_default_graph()
            gc.collect()

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
            with open(self.results_file, 'a') as f:
                json.dump(result_entry, f)
                f.write('\n')

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
                            'win_rate_mcts': float(win_rate_mcts),
                            'mcts_improvement': float(mcts_improvement)
                        },
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                print(f"\n新的最佳得分: {score:.4f}")
            print(f"\n得分详情:")
            print(f"  训练胜率: {train_win_rate:.2%}")
            print(f"  网络胜率: {win_rate_no_mcts:.2%}")
            print(f"  MCTS胜率: {win_rate_mcts:.2%}")
            print(f"  MCTS提升: {mcts_improvement:.2%}")
            print(f"  参数奖励: {param_bonus:.2f}")
            print(f"  最终得分: {score:.4f}")

            return score

        except Exception as e:
            print(f"评估失败: {e}")
            import traceback
            traceback.print_exc()
            return -10.0

    def validate_parameters(self, params):
        """验证参数合理性"""
        if params['pi_lr'] / params['critic_lr'] > 0.5:
            print(f"Policy学习率/Critic学习率比例过高: {params['pi_lr'] / params['critic_lr']:.2f}")
            return False
        if params['batch_size'] < 8 or params['batch_size'] > 128:
            print(f"批次大小超出范围: {params['batch_size']}")
            return False

        if params['rollout_limit'] > params['mcts_simulations']:
            print(f"Rollout限制 > MCTS模拟次数: {params['rollout_limit']} > {params['mcts_simulations']}")
            return False
        return True

    def calculate_param_bonus(self, params):
        """计算参数合理性奖励"""
        bonus = 0
        ratio = params['pi_lr'] / params['critic_lr']
        if 0.05 <= ratio <= 0.2:
            bonus += 5
        elif 0.02 <= ratio <= 0.5:
            bonus += 2

        # 批次大小奖励
        if 16 <= params['batch_size'] <= 64:
            bonus += 3

        # MCTS模拟次数奖励
        if 100 <= params['mcts_simulations'] <= 250:
            bonus += 3
        # 探索权重奖励
        if 0.8 <= params['exploration_weight'] <= 1.5:
            bonus += 2
        # Rollout限制奖励
        if 30 <= params['rollout_limit'] <= 80:
            bonus += 2
        return bonus

    def run_optimization(self, n_iter=25, init_points=8):
        """运行贝叶斯优化"""
        print(f"迭代次数: {n_iter}")
        print(f"初始随机点: {init_points}")
        print(f"训练游戏数/参数: {self.num_train_games}")
        print(f"评估游戏数/参数: {self.num_eval_games}")
        print(f"日志目录: {self.log_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = JSONLogger(path=f"{self.log_dir}/optimization_{timestamp}.json")
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
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
        self.print_final_results(elapsed_time)
        return self.optimizer.max

    def print_final_results(self, elapsed_time):
        """打印最终结果"""
        if hasattr(self.optimizer, 'max'):
            best_result = self.optimizer.max
            print("优化完成!")
            print(f"最佳得分: {best_result['target']:.4f}")
            print("\n最佳参数配置:")
            best_params = self.convert_parameters(best_result['params'])

            print("MCTS参数:")
            print(f"  模拟次数: {best_params['mcts_simulations']}")
            print(f"  探索权重: {best_params['exploration_weight']:.3f}")
            print(f"  Rollout限制: {best_params['rollout_limit']}")

            print("\nRL算法参数:")
            print(f"  Critic学习率: {best_params['critic_lr']:.6f}")
            print(f"  Policy学习率: {best_params['pi_lr']:.6f}")
            print(f"  熵正则化: {best_params['entropy_cost']:.4f}")
            print(f"  批次大小: {best_params['batch_size']}")
            print(f"  Critic更新次数: {best_params['num_critic_before_pi']}")

        print(f"总耗时: {elapsed_time:.1f}秒")
        print(f"测试参数组合数: {len(self.history)}")
        print(f"结果已保存到: {self.results_file}")


def main():
    """主函数"""
    optimizer = MiniAlphaGoOptimizer(
        num_eval_games=15,
        num_train_games=20,
        log_dir="./bayesian_optimization"
    )
    optimizer.run_optimization(
        n_iter=20,  # 迭代次数
        init_points=8  # 初始随机点
    )


if __name__ == "__main__":
    main()