import numpy as np
import sonnet as snt
import tensorflow as tf
from collections import defaultdict


# mcts.py框架
class MCTSNode:
    """MCTS树节点"""
    def __init__(self, state, player, parent=None, prior=0):
        self.state = state
        self.player = player  # 当前玩家 (0=黑, 1=白)
        self.parent = parent
        self.children = {} # 动作->子节点
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # 来自策略网络的先验概率

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, exploration_weight=1.0, simulation_limit=50):
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.root = None

    def _select(self, node):
        # TODO:选择阶段：使用UCB选择子节点
        pass

    def _expand(self, node):
        #TODO:扩展阶段：为当前节点添加子节点
        pass

    def _evaluate(self, state):
        # 使用浅层网络快速走子
        pass

    def _backpropagate(self, node, value):
        #TODO:回传阶段：更新路径上的统计信息
        pass