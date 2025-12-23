import numpy as np
import math
import copy
import os
from environment import coords
from agent.agent import GoPolicyAgent
import tensorflow as tf

class MCTSNode:
    """MCTS树节点"""
    def __init__(self, position, player, parent=None, prior=0.0):
        self.position = position  # go.Position对象
        self.player = player  # 当前玩家 (0=黑, 1=白)
        self.parent = parent
        self.children = {}  # 动作->子节点
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # 来自策略网络的先验概率

    def expanded(self):
        return len(self.children) > 0

    def is_terminal(self):
        return self.position.is_game_over()

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, exploration_weight=1.0, simulation_limit=20):
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.root = None
        self.N = int(os.environ.get('BOARD_SIZE', 5))

    def _select(self, node):
        """选择阶段：使用UCB选择子节点"""
        # UCB公式：Q + c * sqrt(ln(N)/n)
        if not node.expanded() or node.is_terminal():
            return node

        best_score = -float('inf')
        best_child = None

        for action, child in node.children.items():
            # 计算UCB分数
            if child.visit_count == 0:
                ucb_score = float('inf')  # 优先选择未访问的
            else:
                q_value = child.value()
                exploration = math.sqrt(math.log(node.visit_count + 1) / (child.visit_count + 1e-8))
                ucb_score = q_value + self.exploration_weight * child.prior * exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return self._select(best_child)

    def _expand(self, node):
        """扩展阶段：为当前节点添加子节点"""
        legal_moves_mask = node.position.all_legal_moves()
        legal_actions = np.where(legal_moves_mask == 1)[0]

        if len(legal_actions) == 0:
            return

        for action in legal_actions:
            try:
                new_position = copy.deepcopy(node.position)
                if action == self.N * self.N:
                    new_position = new_position.pass_move(mutate=True)
                else:
                    move = coords.from_flat(action)
                    new_position = new_position.play_move(move, mutate=True)

                next_player = 0 if new_position.to_play == 1 else 1

                prior = 1.0 / len(legal_actions)
                child = MCTSNode(new_position, next_player, parent=node, prior=prior)
                node.children[action] = child

            except Exception as e:
                print(f"动作{action}不合法: {e}")
                continue

    def _simulate_random(self, node):
        """随机走子模拟"""
        current_position = copy.deepcopy(node.position)
        steps = 0

        while not current_position.is_game_over() and steps < self.simulation_limit:
            legal_moves_mask = current_position.all_legal_moves()
            legal_actions = np.where(legal_moves_mask == 1)[0]

            if len(legal_actions) == 0:
                break

            action = np.random.choice(legal_actions)

            try:
                if action == self.N * self.N:  # pass动作
                    current_position = current_position.pass_move(mutate=True)
                else:
                    move = coords.from_flat(action)
                    current_position = current_position.play_move(move, mutate=True)

                steps += 1

            except Exception as e:
                print(f"模拟中非法动作: {e}")
                break

        if current_position.is_game_over():
            result = current_position.result()

            if node.player == 0:
                value = result
            else:
                value = -result
        else:
            score = current_position.score()
            max_score = self.N * self.N
            value = np.clip(score / max_score, -1, 1)

            if node.player == 1:
                value = -value

        return value

    def _simulate(self,node):
        return self._simulate_random(node)

    def _backpropagate(self, node, value):
        """回传阶段：更新路径上的统计信息"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value  # 胜负视角转换
            current = current.parent

    def search(self, position, current_player, num_simulations=50):
        """执行MCTS搜索"""
        # 创建根节点,扩展根节点
        self.root = MCTSNode(position, current_player)
        self._expand(self.root)

        for i in range(num_simulations):
            leaf = self._select(self.root) # 1. 选择
            if not leaf.is_terminal() and leaf.visit_count > 0: # 2. 扩展（如果叶子节点被访问过且不是终局）
                self._expand(leaf)
                leaf = self._select(leaf)
            value = self._simulate(leaf) # 3. 模拟
            self._backpropagate(leaf, value) # 4. 回传

        legal_moves_mask = position.all_legal_moves()
        visit_counts = np.zeros(len(legal_moves_mask))

        for action, child in self.root.children.items():
            if 0 <= action < len(visit_counts):
                visit_counts[action] = child.visit_count

        total_visits = sum(visit_counts)
        if total_visits > 0:
            action_probs = visit_counts / total_visits
        else:
            action_probs = legal_moves_mask / (legal_moves_mask.sum() + 1e-8)

        return action_probs

    def get_best_action(self, position, current_player, num_simulations=50, temperature=1.0):
        """根据MCTS搜索结果选择最佳动作"""
        action_probs = self.search(position, current_player, num_simulations)

        legal_moves_mask = position.all_legal_moves()
        legal_actions = np.where(legal_moves_mask == 1)[0]

        if len(legal_actions) == 0:
            return self.N * self.N, action_probs  # 只能pass

        legal_probs = action_probs[legal_actions]
        legal_probs = legal_probs / (legal_probs.sum() + 1e-8)

        if temperature == 0:
            best_idx = np.argmax(legal_probs)
            best_action = legal_actions[best_idx]
        else:
            log_probs = np.log(legal_probs + 1e-8) / temperature
            exp_log_probs = np.exp(log_probs)
            probs = exp_log_probs / exp_log_probs.sum()
            chosen_idx = np.random.choice(len(legal_actions), p=probs)
            best_action = legal_actions[chosen_idx]

        return best_action, action_probs

    def update_root(self, action):
        """更新根节点（用于连续决策）"""
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

class AlphaGoMCTS(MCTS):
    """整合策略网络的AlphaGo MCTS"""
    def __init__(self, deep_policy_agent, rollout_policy_agent,
                 exploration_weight=1.0, simulation_limit=50):
        super().__init__(exploration_weight, simulation_limit)
        tf.reset_default_graph()
        self.deep_sess = tf.Session()

        self.deep_policy = GoPolicyAgent(
            session=self.deep_sess,
            hidden_layers=[256, 256],  # 和训练时相同
            loss_str="a2c"
        )

        self.deep_sess.run(tf.global_variables_initializer())
        if os.path.exists(deep_policy_agent + ".index"):
            self.deep_policy.restore(deep_policy_agent)
            print("✅ 深度网络加载成功")
        else:
            print(f"❌ 找不到深度网络文件: {deep_policy_agent}")

        tf.reset_default_graph()
        self.rollout_sess = tf.Session()

        self.rollout_policy = GoPolicyAgent(
            session=self.rollout_sess,
            hidden_layers=[64],
            loss_str="a2c"
        )

        self.rollout_sess.run(tf.global_variables_initializer())
        if os.path.exists(rollout_policy_agent + ".index"):
            self.rollout_policy.restore(rollout_policy_agent)
            print("✅ 浅层网络加载成功")
        else:
            print(f"❌ 找不到浅层网络文件: {rollout_policy_agent}")

        self.board_size = 5
        print("AlphaGo MCTS 初始化完成")

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
                print(f"网络失败:{e}，退回到随机")
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