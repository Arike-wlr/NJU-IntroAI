import numpy as np
import sonnet as snt
import tensorflow as tf
from collections import defaultdict


# mcts.py框架
class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  # 来自策略网络的先验概率

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, deep_net, shallow_net, num_simulations=100):
        self.deep_net = deep_net  # 树策略
        self.shallow_net = shallow_net  # rollout策略
        self.num_simulations = num_simulations

    def search(self, state):
        # 执行多次模拟
        for _ in range(self.num_simulations):
            self._simulate(state)
        # 返回动作概率

    def _simulate(self, state):
        # 选择 → 扩展 → 评估 → 回传
        pass

    def _select(self, node):
        # PUCT算法选择
        pass

    def _evaluate(self, state):
        # 使用浅层网络快速走子
        pass