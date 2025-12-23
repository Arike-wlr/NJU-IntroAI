import numpy as np
import math
import copy
import os
from environment import coords

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
            value = self._simulate_random(leaf) # 3. 模拟
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